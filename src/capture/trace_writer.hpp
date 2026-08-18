// Three-stream trace writer — plan T2.3.
//
// Layout, strides and file names all come from trace_format.h, which is GENERATED from
// src/traces/format.py. The Python reader and this writer therefore cannot disagree about the
// on-disk format; a layout change is one edit followed by regenerating the header.
//
// Why documents are buffered
// --------------------------
// Both topk.bin and logits.bin are LAYER-MAJOR WITHIN TOKEN, but the callback delivers a whole
// ubatch of ONE layer at a time. So the arrival order is the transpose of the file order and no
// sequential write is possible. Writing at computed offsets instead would mean one tiny pwrite per
// (token, layer) - 98k syscalls per document for Qwen3 - so a document is assembled in RAM in final
// layout and written in one sequential pass per stream at document end.
//
// The buffers are sized once from n_ctx (documents are capped at 2048 tokens by plan T4.2) and
// reused for every document, which is also what plan T2.2 rule 5 requires: no allocation inside
// the callback.
//
//   topk   2048 tok x 48 layers x 8 x 4 B  =  3.1 MB   (Qwen3, the widest)
//   logits 2048 tok x 48 layers x 128 x 2 B = 25.2 MB
//   hidden captured-only x 48 x 2048 x 2 B  = ~20 MB at a 1-in-20 subsample

#pragma once

#include "trace_format.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace moe {

// IEEE-754 binary32 -> binary16, round-to-nearest-even, with correct subnormal and overflow
// handling. Written out rather than pulled from ggml so the trace format has no dependency on
// which ggml build produced it: a trace written with a different rounding rule is a different
// experiment, and this keeps that decision in one visible place.
inline uint16_t f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t) ((bits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = bits & 0x007FFFFFu;

    if (exponent == 128) {  // Inf or NaN
        // Preserve NaN-ness: a quiet NaN must not collapse to Inf, because plan T5.3 checks for
        // NaN/Inf in the trace and the two mean different things.
        return (uint16_t) (sign | 0x7C00u | (mantissa ? 0x0200u : 0u));
    }
    if (exponent > 15) {  // overflow
        return (uint16_t) (sign | 0x7C00u);
    }
    if (exponent >= -14) {  // normal
        uint32_t half = ((uint32_t) (exponent + 15) << 10) | (mantissa >> 13);
        const uint32_t rest = mantissa & 0x1FFFu;
        if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) {
            half++;  // ties-to-even; carries into the exponent correctly
        }
        return (uint16_t) (sign | half);
    }
    if (exponent >= -24) {  // subnormal
        mantissa |= 0x00800000u;
        const int shift = -exponent - 14 + 13;
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1);
        if (rest > midpoint || (rest == midpoint && (half & 1u))) {
            half++;
        }
        return (uint16_t) (sign | half);
    }
    return (uint16_t) sign;  // underflow to signed zero
}

struct stream_file {
    FILE * fp = nullptr;
    std::string path;
    uint64_t bytes = 0;

    bool open(const std::string & dir, const char * name) {
        path = dir + "/" + name;
        fp = fopen(path.c_str(), "wb");
        return fp != nullptr;
    }
    bool write(const void * data, size_t n) {
        if (n == 0) return true;
        if (fwrite(data, 1, n, fp) != n) return false;
        bytes += n;
        return true;
    }
    bool close() {
        if (!fp) return true;
        const bool ok = fflush(fp) == 0;
        const bool closed = fclose(fp) == 0;
        fp = nullptr;
        return ok && closed;
    }
};

// Create `dir` and any missing parents. Already existing is success, not an error -- resuming a
// shard (plan S.3) re-opens a directory that is already there.
inline bool ensure_directory(const std::string & dir, std::string & err) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(dir), ec);
    if (ec) {
        err = "cannot create output directory " + dir + ": " + ec.message();
        return false;
    }
    if (!std::filesystem::is_directory(std::filesystem::path(dir), ec)) {
        err = "output path exists but is not a directory: " + dir;
        return false;
    }
    return true;
}

struct trace_writer {
    // geometry
    int n_moe_layers = 0;
    int n_experts = 0;
    int top_k = 0;
    int hidden_dim = 0;
    int max_doc_tokens = 0;

    // files
    stream_file f_tokens, f_topk, f_logits, f_hidden, f_hidden_index;

    // per-document staging, in final on-disk layout
    std::vector<moe_token_record> buf_tokens;
    std::vector<int32_t>  buf_topk;    // [tok][layer][top_k]
    std::vector<uint16_t> buf_logits;  // [tok][layer][n_experts]
    std::vector<uint16_t> buf_hidden;  // [captured][layer][hidden_dim]
    std::vector<uint32_t> buf_hidden_index;
    std::vector<int32_t>  hidden_slot;  // doc-relative token -> row in buf_hidden, or -1

    int doc_tokens = 0;
    int doc_captured = 0;

    // corpus-wide totals, for the stats report and the manifest Python writes
    uint64_t total_tokens = 0;
    uint64_t total_captured = 0;

    std::string error;

    bool ok() const { return error.empty(); }

    size_t topk_row() const { return (size_t) top_k; }
    size_t logit_row() const { return (size_t) n_experts; }
    size_t hidden_row() const { return (size_t) hidden_dim; }

    bool open(const std::string & dir, int layers, int experts, int k, int hdim, int max_tokens) {
        n_moe_layers = layers;
        n_experts = experts;
        top_k = k;
        hidden_dim = hdim;
        max_doc_tokens = max_tokens;

        // Shard directories are named from the model and shard id (plan S.3) and never pre-exist on
        // a fresh session, so the writer creates its own. Doing this here rather than in the caller
        // keeps it on the one path that is about to write into it -- a collection run that failed
        // for want of an mkdir would burn a GPU-hour of Kaggle quota to report a missing directory.
        if (!ensure_directory(dir, error)) {
            return false;
        }

        if (!f_tokens.open(dir, MOE_FILE_TOKENS) || !f_topk.open(dir, MOE_FILE_TOPK) ||
            !f_logits.open(dir, MOE_FILE_LOGITS) || !f_hidden.open(dir, MOE_FILE_HIDDEN) ||
            !f_hidden_index.open(dir, MOE_FILE_HIDDEN_INDEX)) {
            error = "cannot create stream files under " + dir;
            return false;
        }

        buf_tokens.resize((size_t) max_doc_tokens);
        buf_topk.assign((size_t) max_doc_tokens * n_moe_layers * top_k, 0);
        buf_logits.assign((size_t) max_doc_tokens * n_moe_layers * n_experts, 0);
        buf_hidden_index.reserve((size_t) max_doc_tokens);
        hidden_slot.assign((size_t) max_doc_tokens, -1);
        return true;
    }

    // Called once per document, before prefill. `capture` marks which doc-relative tokens get a
    // router-input vector; the hidden buffer is sized to exactly that count.
    void begin_document(int n_tokens, const std::vector<bool> & capture) {
        doc_tokens = n_tokens;
        doc_captured = 0;
        std::fill(hidden_slot.begin(), hidden_slot.begin() + n_tokens, -1);
        for (int i = 0; i < n_tokens; ++i) {
            if (capture[(size_t) i]) {
                hidden_slot[(size_t) i] = doc_captured++;
            }
        }
        buf_hidden.assign((size_t) doc_captured * n_moe_layers * hidden_dim, 0);
        buf_hidden_index.clear();
    }

    size_t topk_offset(int token, int layer) const {
        return ((size_t) token * n_moe_layers + layer) * topk_row();
    }
    size_t logit_offset(int token, int layer) const {
        return ((size_t) token * n_moe_layers + layer) * logit_row();
    }
    size_t hidden_offset(int slot, int layer) const {
        return ((size_t) slot * n_moe_layers + layer) * hidden_row();
    }

    // -- per-ubatch ingestion -------------------------------------------------------------------
    //
    // `src` is the de-strided row block for this (stream, layer): n_rows rows of n_cols elements.
    // `first_token` is the doc-relative index of row 0.

    void put_topk(int layer, int first_token, int64_t n_rows, const int32_t * src) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const int token = first_token + (int) r;
            std::memcpy(&buf_topk[topk_offset(token, layer)], src + r * top_k,
                        topk_row() * sizeof(int32_t));
        }
    }

    void put_logits(int layer, int first_token, int64_t n_rows, const float * src) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const int token = first_token + (int) r;
            uint16_t * dst = &buf_logits[logit_offset(token, layer)];
            const float * row = src + r * n_experts;
            for (int e = 0; e < n_experts; ++e) {
                dst[e] = f32_to_f16(row[e]);
            }
        }
    }

    void put_router_input(int layer, int first_token, int64_t n_rows, const float * src) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const int token = first_token + (int) r;
            const int slot = hidden_slot[(size_t) token];
            if (slot < 0) continue;  // not in the subsample
            uint16_t * dst = &buf_hidden[hidden_offset(slot, layer)];
            const float * row = src + r * hidden_dim;
            for (int d = 0; d < hidden_dim; ++d) {
                dst[d] = f32_to_f16(row[d]);
            }
        }
    }

    // -- per-document flush ----------------------------------------------------------------------

    bool end_document(uint32_t doc_id, const int32_t * token_ids, uint64_t global_token_base) {
        for (int i = 0; i < doc_tokens; ++i) {
            moe_token_record & rec = buf_tokens[(size_t) i];
            rec.token_id = (uint32_t) token_ids[i];
            rec.doc_id = doc_id;
            rec.pos_in_doc = (uint32_t) i;
            rec.flags = hidden_slot[(size_t) i] >= 0 ? MOE_FLAG_HIDDEN_CAPTURED : 0u;
            if (hidden_slot[(size_t) i] >= 0) {
                // GLOBAL, corpus-relative index, so shards concatenate without rewriting (T2.3).
                buf_hidden_index.push_back((uint32_t) (global_token_base + (uint64_t) i));
            }
        }

        const bool wrote =
            f_tokens.write(buf_tokens.data(), (size_t) doc_tokens * sizeof(moe_token_record)) &&
            f_topk.write(buf_topk.data(),
                         (size_t) doc_tokens * n_moe_layers * topk_row() * sizeof(int32_t)) &&
            f_logits.write(buf_logits.data(),
                           (size_t) doc_tokens * n_moe_layers * logit_row() * sizeof(uint16_t)) &&
            f_hidden.write(buf_hidden.data(), buf_hidden.size() * sizeof(uint16_t)) &&
            f_hidden_index.write(buf_hidden_index.data(),
                                 buf_hidden_index.size() * sizeof(uint32_t));
        if (!wrote) {
            error = "short write flushing document " + std::to_string(doc_id) +
                    " (disk full, or the scratch quota was hit mid-shard)";
            return false;
        }

        total_tokens += (uint64_t) doc_tokens;
        total_captured += (uint64_t) doc_captured;
        return true;
    }

    // Size arithmetic must hold exactly; plan T5.3 re-checks it from Python, and check_file_sizes()
    // in src/traces/format.py raises on any byte mismatch. Checking here too means the failure is
    // caught before the shard is uploaded rather than after.
    bool close_and_verify() {
        const bool closed = f_tokens.close() && f_topk.close() && f_logits.close() &&
                            f_hidden.close() && f_hidden_index.close();
        if (!closed) {
            error = "failed to close one or more stream files";
            return false;
        }

        struct expect { const char * name; uint64_t got; uint64_t want; };
        const expect checks[] = {
            { MOE_FILE_TOKENS, f_tokens.bytes, total_tokens * MOE_TOKEN_RECORD_BYTES },
            { MOE_FILE_TOPK, f_topk.bytes, total_tokens * MOE_TOPK_STRIDE(n_moe_layers, top_k) },
            { MOE_FILE_LOGITS, f_logits.bytes,
              total_tokens * MOE_LOGIT_STRIDE(n_moe_layers, n_experts) },
            { MOE_FILE_HIDDEN, f_hidden.bytes,
              total_captured * MOE_HIDDEN_STRIDE(n_moe_layers, hidden_dim) },
            { MOE_FILE_HIDDEN_INDEX, f_hidden_index.bytes,
              total_captured * MOE_HIDDEN_INDEX_ELEM_BYTES },
        };
        for (const expect & c : checks) {
            if (c.got != c.want) {
                error = std::string(c.name) + ": wrote " + std::to_string(c.got) +
                        " bytes, size arithmetic expects " + std::to_string(c.want);
                return false;
            }
        }
        return true;
    }
};

}  // namespace moe
