// moe_trace -- capture MoE routing traces from a GGUF model -- plan T2.1 / T2.2.
//
// Standalone binary against the public llama.cpp API only (llama.h + ggml.h). No llama.cpp source
// is patched, so a llama.cpp bump is a rebuild rather than a re-port.
//
// Usage:
//   moe_trace --model M.gguf --spec nodes.spec --corpus shard.jsonl --out shard_00007/ \
//             --ctx 2048 --ubatch 512 --ngl 99 --hidden-stride 20 --shard-id 7
//
// It reads a JSONL shard where each line is {"doc_id": <int>, "text": "..."}, prefills each
// document with a cleared KV cache, and writes the five binary streams. The manifest is written by
// the Python runner, which owns the SHA256s and the run-config identity (plan S.3).
//
// WHY DOCUMENT-LEVEL SHARDING IS BIT-EXACT: each document is prefilled from an empty KV cache, so a
// document's routing depends on nothing outside itself. Shards can therefore be collected in any
// order, in any session, and concatenated -- provided run_config_sha256 matches, which is the check
// src/runtime/config.py enforces as a hard error.
//
// -----------------------------------------------------------------------------------------------
// The six correctness rules in the callback. Five are plan T2.2; the sixth was found by reading
// llama.cpp master and is not in the plan.
//
//  1. Read t->ne[1], the ubatch token count. Reading column 0 only would capture one token per
//     ubatch and silently discard the other 511.
//  2. ggml_backend_tensor_get(), never raw t->data. The buffer may not be host-visible, and with
//     -sm layer across two T4s the same layer lands on a different device between models.
//  3. Assert t->type per stream. topk is I32; casting it to const float* yields garbage silently.
//  4. No stderr/printf inside the callback. Errors are recorded in the state and reported by main.
//  5. Scratch grows monotonically and is never freed inside the callback.
//  6. *** ffn_moe_topk IS A STRIDED VIEW. *** ggml_argsort_top_k() is
//
//         result = ggml_argsort(ctx, a, DESC);                       // [n_expert,      n_tokens]
//         result = ggml_view_4d(ctx, result, k, ne1, ne2, ne3,
//                               result->nb[1], ...);                 // [n_expert_used, n_tokens]
//
//     so ne[0] == top_k while nb[1] == n_expert * 4. The rows are NOT packed. A contiguous read of
//     top_k * n_tokens * 4 bytes returns token 0's first top_k experts, then the REST OF TOKEN 0,
//     and so on -- valid, in-range, distinct expert indices that are wrong for every token but the
//     first. It would pass T5.3's range and distinctness checks and corrupt the entire study
//     silently. Every stream is therefore de-strided through t->nb[1] unconditionally, and the
//     observed layout is recorded so the operator can see which one occurred.
//
//     Note ggml_top_k() (GGML_OP_TOP_K) returns a contiguous tensor, and older llama.cpp used it
//     here. Which one build_moe_ffn calls is therefore version-dependent, which makes the pinned
//     llama_cpp_commit load-bearing for CORRECTNESS, not just reproducibility.
// -----------------------------------------------------------------------------------------------

#include "llama.h"
#include "ggml.h"

#include "node_spec.hpp"
#include "trace_writer.hpp"

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------
// callback state
// ---------------------------------------------------------------------------------------------

struct capture_state {
    moe::node_spec spec;
    moe::trace_writer * writer = nullptr;

    // Rule 5: one scratch buffer, grown monotonically, never freed in the callback.
    std::vector<uint8_t> scratch;

    // Row cursors, one per (stream, trace layer). A cursor is the doc-relative index of the next
    // row this stream expects, so it doubles as "where does this ubatch start" and as the T3.1
    // lockstep check at document end.
    std::vector<int64_t> cursor_topk;
    std::vector<int64_t> cursor_logits;
    std::vector<int64_t> cursor_hidden;

    int doc_tokens = 0;

    // diagnostics, reported by main (rule 4)
    bool failed = false;
    std::string failure;
    int64_t n_nodes_captured = 0;
    bool saw_strided_topk = false;
    bool saw_contiguous_topk = false;
    bool capture_enabled = true;

    void reset_cursors() {
        std::fill(cursor_topk.begin(), cursor_topk.end(), 0);
        std::fill(cursor_logits.begin(), cursor_logits.end(), 0);
        std::fill(cursor_hidden.begin(), cursor_hidden.end(), 0);
    }

    void fail(const std::string & msg) {
        if (!failed) {
            failed = true;
            failure = msg;
        }
    }
};

// De-strided read of a node's rows into scratch, returning a pointer to packed row data.
//
// Reads ggml_nbytes(t) -- which for a strided view is the full SPAN including the discarded
// columns -- then compacts rows by t->nb[1]. Correct for both contiguous and strided tensors, so
// there is no layout branch that can be wrong for one model and right for another.
const void * read_rows_packed(capture_state * st, ggml_tensor * t, int64_t n_cols, int64_t n_rows,
                              size_t elem_size) {
    const size_t span = ggml_nbytes(t);
    const size_t packed = (size_t) n_cols * (size_t) n_rows * elem_size;

    if (st->scratch.size() < span + packed) {
        st->scratch.resize(span + packed);  // grows only; rule 5
    }
    uint8_t * raw = st->scratch.data();
    uint8_t * out = raw + span;

    ggml_backend_tensor_get(t, raw, 0, span);  // rule 2

    const size_t row_bytes = (size_t) n_cols * elem_size;
    const size_t src_stride = t->nb[1];

    if (src_stride == row_bytes) {
        return raw;  // already packed; skip the copy
    }
    for (int64_t r = 0; r < n_rows; ++r) {
        std::memcpy(out + (size_t) r * row_bytes, raw + (size_t) r * src_stride, row_bytes);
    }
    return out;
}

bool moe_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    // T0.5's control leg. With `capture_enabled` false the callback is still installed and the
    // scheduler still runs its callback path, but nothing is ever requested, so the graph is
    // computed in whole splits and no tensor is read back. The difference between this and a full
    // capture is the cost of *requesting* tensors -- the extra ggml_backend_synchronize() per
    // requested node and the device-to-host copies -- which is what the plan asks us to quantify
    // and what llama-bench structurally cannot measure.
    if (!st->capture_enabled) {
        return true;
    }

    const moe::node_target * target = st->spec.find(t->name);

    if (ask) {
        // Phase 1: the scheduler asks whether we need this node's data. Saying no lets it batch
        // the node with its neighbours; saying yes makes it the last node of a single-node graph
        // view, which is what guarantees the tensor is materialised even when the CUDA backend
        // would otherwise fuse the top-k chain into one kernel.
        return target != nullptr;
    }
    if (target == nullptr || st->failed) {
        return !st->failed;
    }

    const int64_t n_cols = t->ne[0];
    const int64_t n_rows = t->ne[1];  // rule 1

    int64_t * cursor = nullptr;
    int64_t expect_cols = 0;
    ggml_type expect_type = GGML_TYPE_F32;
    const char * label = "";

    switch (target->kind) {
        case moe::stream_kind::topk:
            cursor = &st->cursor_topk[(size_t) target->trace_layer];
            expect_cols = st->spec.top_k;
            expect_type = GGML_TYPE_I32;  // rule 3
            label = "topk";
            break;
        case moe::stream_kind::logits:
            cursor = &st->cursor_logits[(size_t) target->trace_layer];
            expect_cols = st->spec.n_experts;
            label = "logits";
            break;
        case moe::stream_kind::router_input:
            cursor = &st->cursor_hidden[(size_t) target->trace_layer];
            expect_cols = st->spec.hidden_dim;
            label = "router_input";
            break;
        default:
            return true;
    }

    if (t->type != expect_type) {
        st->fail(std::string("stream ") + label + " node " + t->name + " has ggml type " +
                 ggml_type_name(t->type) + ", expected " + ggml_type_name(expect_type) +
                 " (plan T2.2 rule 3)");
        return false;
    }
    if (n_cols != expect_cols) {
        st->fail(std::string("stream ") + label + " node " + t->name + " has ne[0]=" +
                 std::to_string(n_cols) + ", spec says " + std::to_string(expect_cols) +
                 "; the node spec does not match this model (plan T1.4)");
        return false;
    }
    if (*cursor + n_rows > st->doc_tokens) {
        st->fail(std::string("stream ") + label + " node " + t->name + " would write row " +
                 std::to_string(*cursor + n_rows) + " of a " + std::to_string(st->doc_tokens) +
                 "-token document; the graph produced more rows than tokens");
        return false;
    }

    if (target->kind == moe::stream_kind::topk) {
        const bool strided = t->nb[1] != (size_t) n_cols * ggml_type_size(t->type);
        (strided ? st->saw_strided_topk : st->saw_contiguous_topk) = true;  // rule 6 diagnostic
    }

    const size_t elem = ggml_type_size(t->type);
    const void * rows = read_rows_packed(st, t, n_cols, n_rows, elem);
    const int first = (int) *cursor;

    switch (target->kind) {
        case moe::stream_kind::topk:
            st->writer->put_topk(target->trace_layer, first, n_rows, (const int32_t *) rows);
            break;
        case moe::stream_kind::logits:
            st->writer->put_logits(target->trace_layer, first, n_rows, (const float *) rows);
            break;
        case moe::stream_kind::router_input:
            st->writer->put_router_input(target->trace_layer, first, n_rows, (const float *) rows);
            break;
        default:
            break;
    }

    *cursor += n_rows;
    st->n_nodes_captured++;
    return true;
}

// ---------------------------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------------------------

// T0.5 needs three legs, not the two the plan sketches. "Filter returns false for everything"
// isolates the cost of reading tensors back, but `cb_eval` is still set, and setting it changes how
// ggml_backend_sched executes a graph regardless of what the callback answers. Separating
// `no_callback` from `filter_off` splits that into "the cost of being observable at all" and "the
// cost of actually observing", and the two have different remedies if the ratio is bad.
enum class capture_mode {
    full,        // normal collection: request nodes, read them, write the trace
    filter_off,  // cb_eval installed, nothing requested, nothing written
    no_callback, // cb_eval not installed at all -- the true baseline
};

struct options {
    std::string model_path;
    std::string spec_path;
    std::string corpus_path;
    std::string out_dir;
    std::string stats_path;
    std::string tensor_split;
    int n_ctx = 2048;    // plan I4: ALWAYS pinned. Gemma 4 defaults to 262144.
    int n_batch = 2048;
    int n_ubatch = 512;
    int n_gpu_layers = 99;
    int n_threads = 4;
    int hidden_stride = 20;
    int shard_id = 0;

    bool flash_attn = true;
    bool split_mode_row = false;
    capture_mode mode = capture_mode::full;

    bool writes_trace() const { return mode == capture_mode::full; }
};

bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string & dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        auto next_int = [&](int & dst) {
            std::string raw;
            if (!next(raw)) return false;
            dst = std::atoi(raw.c_str());
            return true;
        };
        bool ok = true;
        if (a == "--model") ok = next(o.model_path);
        else if (a == "--spec") ok = next(o.spec_path);
        else if (a == "--corpus") ok = next(o.corpus_path);
        else if (a == "--out") ok = next(o.out_dir);
        else if (a == "--stats") ok = next(o.stats_path);
        else if (a == "--ctx") ok = next_int(o.n_ctx);
        else if (a == "--batch") ok = next_int(o.n_batch);
        else if (a == "--ubatch") ok = next_int(o.n_ubatch);
        else if (a == "--ngl") ok = next_int(o.n_gpu_layers);
        else if (a == "--threads") ok = next_int(o.n_threads);
        else if (a == "--hidden-stride") ok = next_int(o.hidden_stride);
        else if (a == "--shard-id") ok = next_int(o.shard_id);
        else if (a == "--tensor-split") ok = next(o.tensor_split);
        else if (a == "--split-mode-row") o.split_mode_row = true;
        else if (a == "--capture-mode") {
            std::string raw;
            if (!next(raw)) { ok = false; }
            else if (raw == "full")         o.mode = capture_mode::full;
            else if (raw == "filter-off")   o.mode = capture_mode::filter_off;
            else if (raw == "no-callback")  o.mode = capture_mode::no_callback;
            else {
                fprintf(stderr, "moe_trace: --capture-mode must be full, filter-off or "
                                "no-callback (got %s)\n", raw.c_str());
                return false;
            }
        }
        else if (a == "--no-flash-attn") o.flash_attn = false;
        else if (a == "--global-token-base") {
            // Removed deliberately, and rejected rather than ignored: a runner still passing it
            // was built for the old cumulative-reference-token index, and silently accepting the
            // flag would produce a shard whose hidden_index overlaps its neighbours' -- readable,
            // wrong, and only detectable once every shard is concatenated.
            fprintf(stderr,
                    "moe_trace: --global-token-base was removed. The hidden-state index is now "
                    "doc_id*n_ctx + pos_in_doc, a pure function of the corpus; update the "
                    "runner.\n");
            return false;
        } else {
            fprintf(stderr, "moe_trace: unknown argument %s\n", a.c_str());
            return false;
        }
        if (!ok) {
            fprintf(stderr, "moe_trace: %s needs a value\n", a.c_str());
            return false;
        }
    }
    if (o.model_path.empty() || o.spec_path.empty() || o.corpus_path.empty()) {
        fprintf(stderr, "moe_trace: --model, --spec and --corpus are required\n");
        return false;
    }
    if (o.writes_trace() && o.out_dir.empty()) {
        fprintf(stderr, "moe_trace: --out is required unless --capture-mode is a timing mode\n");
        return false;
    }
    if (o.stats_path.empty()) {
        if (o.out_dir.empty()) {
            // A timing run with neither --out nor --stats would produce a measurement and then
            // throw it away. The measurement is the entire point of the run.
            fprintf(stderr, "moe_trace: --stats is required when --capture-mode is a timing mode "
                            "and --out is not given\n");
            return false;
        }
        o.stats_path = o.out_dir + "/capture_stats.json";
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------------------------

struct document {
    uint32_t doc_id = 0;
    std::string text;
};

// Minimal extractor for {"doc_id": N, "text": "..."} with standard JSON string escapes. A full JSON
// parser is not worth a dependency here: the corpus is written by our own Phase 4 code, and a line
// this does not understand is a hard error rather than a silent skip.
bool parse_jsonl_line(const std::string & line, document & doc, std::string & err) {
    const size_t id_at = line.find("\"doc_id\"");
    const size_t text_at = line.find("\"text\"");
    if (id_at == std::string::npos || text_at == std::string::npos) {
        err = "line has no doc_id/text field";
        return false;
    }
    size_t p = line.find(':', id_at);
    if (p == std::string::npos) { err = "malformed doc_id"; return false; }
    doc.doc_id = (uint32_t) std::strtoul(line.c_str() + p + 1, nullptr, 10);

    p = line.find('"', line.find(':', text_at));
    if (p == std::string::npos) { err = "malformed text field"; return false; }
    ++p;

    doc.text.clear();
    while (p < line.size() && line[p] != '"') {
        if (line[p] == '\\' && p + 1 < line.size()) {
            const char c = line[++p];
            switch (c) {
                case 'n': doc.text += '\n'; break;
                case 't': doc.text += '\t'; break;
                case 'r': doc.text += '\r'; break;
                case 'b': doc.text += '\b'; break;
                case 'f': doc.text += '\f'; break;
                case '/': doc.text += '/'; break;
                case '"': doc.text += '"'; break;
                case '\\': doc.text += '\\'; break;
                case 'u': {
                    if (p + 4 >= line.size()) { err = "truncated \\u escape"; return false; }
                    const std::string hex = line.substr(p + 1, 4);
                    unsigned cp = (unsigned) std::strtoul(hex.c_str(), nullptr, 16);
                    p += 4;

                    // Surrogate pairs MUST be recombined. The corpus is written with
                    // ensure_ascii=True (the C++ side is byte-oriented, so ASCII is the safe
                    // interchange), which means every non-BMP character -- emoji, CJK ext-B, and a
                    // good deal of what the 30% multilingual share will drag in -- arrives as a
                    // surrogate pair. Encoding each half independently produces CESU-8: two
                    // 3-byte sequences that no tokenizer recognises. It is not a parse error, so it
                    // would have become silent per-document mojibake, biasing exactly the domain
                    // whose routing behaviour T9.4 is trying to measure.
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // High surrogate: the low half must follow immediately as another \u.
                        if (p + 6 >= line.size() || line[p + 1] != '\\' || line[p + 2] != 'u') {
                            err = "unpaired high surrogate in \\u escape";
                            return false;
                        }
                        const std::string hex_lo = line.substr(p + 3, 4);
                        const unsigned lo = (unsigned) std::strtoul(hex_lo.c_str(), nullptr, 16);
                        if (lo < 0xDC00 || lo > 0xDFFF) {
                            err = "high surrogate not followed by a low surrogate";
                            return false;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p += 6;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        err = "unpaired low surrogate in \\u escape";
                        return false;
                    }

                    if (cp < 0x80) {
                        doc.text += (char) cp;
                    } else if (cp < 0x800) {
                        doc.text += (char) (0xC0 | (cp >> 6));
                        doc.text += (char) (0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        doc.text += (char) (0xE0 | (cp >> 12));
                        doc.text += (char) (0x80 | ((cp >> 6) & 0x3F));
                        doc.text += (char) (0x80 | (cp & 0x3F));
                    } else {
                        doc.text += (char) (0xF0 | (cp >> 18));
                        doc.text += (char) (0x80 | ((cp >> 12) & 0x3F));
                        doc.text += (char) (0x80 | ((cp >> 6) & 0x3F));
                        doc.text += (char) (0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    err = std::string("unsupported escape \\") + c;
                    return false;
            }
            ++p;
        } else {
            doc.text += line[p++];
        }
    }
    if (p >= line.size()) { err = "unterminated text string"; return false; }
    return true;
}

bool load_corpus(const std::string & path, std::vector<document> & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open corpus " + path; return false; }

    std::string line;
    int lineno = 0;
    int c;
    auto flush_line = [&]() -> bool {
        lineno++;
        if (moe::trim(line).empty()) return true;
        document doc;
        std::string why;
        if (!parse_jsonl_line(line, doc, why)) {
            err = path + ":" + std::to_string(lineno) + ": " + why;
            return false;
        }
        out.push_back(std::move(doc));
        return true;
    };

    bool ok = true;
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') {
            if (!(ok = flush_line())) break;
            line.clear();
        } else {
            line += (char) c;
        }
    }
    if (ok && !line.empty()) ok = flush_line();
    fclose(f);

    if (ok && out.empty()) {
        err = path + ": no documents";
        return false;
    }
    return ok;
}

}  // namespace

// ---------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    options opt;
    if (!parse_args(argc, argv, opt)) return 2;

    capture_state st;
    st.spec = moe::load_node_spec(opt.spec_path.c_str());
    if (!st.spec.ok()) {
        fprintf(stderr, "moe_trace: %s\n", st.spec.error.c_str());
        return 2;
    }

    std::vector<document> corpus;
    std::string err;
    if (!load_corpus(opt.corpus_path, corpus, err)) {
        fprintf(stderr, "moe_trace: %s\n", err.c_str());
        return 2;
    }

    st.capture_enabled = opt.writes_trace();

    moe::trace_writer writer;
    if (opt.writes_trace()) {
        if (!writer.open(opt.out_dir, st.spec.n_moe_layers, st.spec.n_experts, st.spec.top_k,
                         st.spec.hidden_dim, opt.n_ctx)) {
            fprintf(stderr, "moe_trace: %s\n", writer.error.c_str());
            return 2;
        }
    }
    st.writer = &writer;
    st.cursor_topk.assign((size_t) st.spec.n_moe_layers, 0);
    st.cursor_logits.assign((size_t) st.spec.n_moe_layers, 0);
    st.cursor_hidden.assign((size_t) st.spec.n_moe_layers, 0);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opt.n_gpu_layers;
    mparams.split_mode = opt.split_mode_row ? LLAMA_SPLIT_MODE_ROW : LLAMA_SPLIT_MODE_LAYER;
    std::vector<float> splits;
    if (!opt.tensor_split.empty()) {
        // Pinned, never auto: an auto split depends on what else is resident on the card, so two
        // sessions could place the same layer on different devices (plan T0.4 / T8.5).
        size_t pos = 0;
        while (pos <= opt.tensor_split.size()) {
            const size_t comma = opt.tensor_split.find(',', pos);
            splits.push_back(std::strtof(opt.tensor_split.substr(pos, comma - pos).c_str(), nullptr));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        mparams.tensor_split = splits.data();
    }

    llama_model * model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "moe_trace: cannot load model %s\n", opt.model_path.c_str());
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) opt.n_ctx;
    cparams.n_batch = (uint32_t) opt.n_batch;
    cparams.n_ubatch = (uint32_t) opt.n_ubatch;
    cparams.n_threads = opt.n_threads;
    cparams.n_threads_batch = opt.n_threads;
    cparams.flash_attn_type = opt.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.embeddings = false;
    if (opt.mode != capture_mode::no_callback) {
        cparams.cb_eval = moe_cb;
        cparams.cb_eval_user_data = &st;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "moe_trace: cannot create context\n");
        llama_model_free(model);
        return 3;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<llama_token> tokens;
    std::vector<int32_t> token_ids;
    std::vector<bool> capture;
    int rc = 0;
    uint64_t n_docs_done = 0;
    uint64_t n_docs_truncated = 0;
    uint64_t n_tokens_dropped = 0;
    uint32_t first_truncated_doc = 0;
    bool any_truncated = false;

    // One allocation for the whole shard: n_ctx is pinned (I4) and every document fits by
    // construction, so re-initialising per document would only add churn.
    llama_batch batch = llama_batch_init(opt.n_ctx, /*embd=*/0, /*n_seq_max=*/1);

    // Decode time only. Model load is excluded deliberately: it is paid once per session and is
    // dominated by disk, so folding it in would make a 20k-token calibration run look slower than
    // the 1M-token collection it is meant to predict.
    double decode_seconds = 0.0;
    uint64_t n_tokens_decoded = 0;

    for (const document & doc : corpus) {
        tokens.resize((size_t) opt.n_ctx);
        int n = llama_tokenize(vocab, doc.text.c_str(), (int32_t) doc.text.size(), tokens.data(),
                               opt.n_ctx, /*add_special=*/true, /*parse_special=*/false);
        if (n < 0) {
            // Truncation, not an error: plan T4.2 caps documents at 2048 tokens, and -n is the
            // count that WOULD have been produced.
            //
            // But it must never be SILENT. T4.2's cap is enforced under one reference tokenizer,
            // while the panel's vocabularies span 50k to 262k, so the same document is a different
            // token count for every checkpoint. A document at the cap for OLMoE can exceed it for
            // Gemma 4, and then the two models were not shown the same text -- which is the one
            // assumption every cross-model comparison in Phase 9 rests on. Counting it here turns a
            // corpus-construction bug into a number in capture_stats.json that T5.3 can check,
            // rather than an unexplained divergence found during analysis.
            n_docs_truncated++;
            n_tokens_dropped += (uint64_t) (-n - opt.n_ctx);
            if (!any_truncated) {
                first_truncated_doc = doc.doc_id;
                any_truncated = true;
            }
            n = opt.n_ctx;
        }
        if (n == 0) {
            fprintf(stderr, "moe_trace: doc %u tokenized to zero tokens\n", doc.doc_id);
            rc = 4;
            break;
        }
        tokens.resize((size_t) n);

        // The global index of a token is `doc_id * n_ctx + pos_in_doc` -- NOT a running count.
        //
        // A running count has to start somewhere, and a session resuming at shard 12 cannot know
        // how many tokens shards 0..11 produced without tokenizing them. The old answer was to
        // seed it with the cumulative *reference* token estimate, which is stable across sessions
        // but is not a token count: consecutive shards then overlap or leave gaps, hidden_index
        // stops being globally ascending, and the reader refuses to concatenate the shard set
        // (T2.3). That is exactly what a real three-shard OLMoE collection produced.
        //
        // Reserving n_ctx indices per document fixes it by construction. Documents are capped at
        // n_ctx tokens (the truncation branch above enforces it), so the blocks cannot collide;
        // the index is a pure function of the corpus, so it is identical in every session and
        // under any re-sharding, which is what T3.6's resume acceptance needs; and it ascends
        // strictly in corpus order, which is what the reader needs. The space is sparse rather
        // than contiguous, and nothing downstream reads it as a token count.
        const uint64_t doc_base = (uint64_t) doc.doc_id * (uint64_t) opt.n_ctx;

        capture.assign((size_t) n, false);
        if (opt.hidden_stride > 0) {
            for (int i = 0; i < n; ++i) {
                // Keyed on the GLOBAL index so the subsample is a property of the corpus position,
                // not of how documents happened to be grouped into shards. Re-sharding therefore
                // does not move which tokens have hidden states.
                capture[(size_t) i] = ((doc_base + (uint64_t) i) % (uint64_t) opt.hidden_stride) == 0;
            }
        }

        token_ids.assign(tokens.begin(), tokens.end());
        if (opt.writes_trace()) writer.begin_document(n, capture);
        st.doc_tokens = n;
        st.reset_cursors();

        // Cleared between documents, which is what makes document-level sharding bit-exact.
        llama_memory_clear(mem, /*data=*/true);

        // Every token must be an OUTPUT token, and this is not an optimisation knob.
        //
        // llama.cpp prunes the final layer to the output rows before the FFN runs:
        //
        //     if (il == n_layer - 1 && inp_out_ids) {
        //         cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
        //         inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        //     }
        //     ... build_norm -> cb "ffn_norm" -> build_moe_ffn
        //
        // (src/models/olmoe.cpp:124, and the same shape in every other decoder). With the default
        // `llama_batch_get_one` batch, logits==nullptr means "last token only", so the last MoE
        // layer routes ONE row per document while every other layer routes n. The deepest layer is
        // the one Phase 9 reads depth trends off, so losing it is not a partial result -- it is a
        // trend computed over a truncated stack. T3.1's lockstep assertion is what caught it.
        //
        // The cost is the lm_head matmul over all n tokens instead of one, which is exactly what
        // llama-perplexity pays; it buys the layer back, and nothing cheaper does.
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = tokens[(size_t) i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        const auto t_decode0 = std::chrono::steady_clock::now();
        const int32_t decoded = llama_decode(ctx, batch);
        decode_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode0).count();
        n_tokens_decoded += (uint64_t) n;
        if (decoded != 0) {
            fprintf(stderr, "moe_trace: llama_decode returned %d on doc %u\n", decoded, doc.doc_id);
            rc = 5;
            break;
        }
        if (st.failed) {
            fprintf(stderr, "moe_trace: capture failed on doc %u: %s\n", doc.doc_id,
                    st.failure.c_str());
            rc = 6;
            break;
        }

        // Three-stream lockstep, per document (plan T3.1). Every stream must have received exactly
        // one row per token for every MoE layer. A missing node, a fused kernel that skipped a
        // tensor, or a spec naming a node that does not exist all land here instead of producing a
        // short file that only size arithmetic would catch later.
        for (int l = 0; opt.writes_trace() && l < st.spec.n_moe_layers; ++l) {
            const int64_t got[3] = { st.cursor_topk[(size_t) l], st.cursor_logits[(size_t) l],
                                     st.cursor_hidden[(size_t) l] };
            const char * names[3] = { "topk", "logits", "router_input" };
            for (int s = 0; s < 3; ++s) {
                if (got[s] != n) {
                    fprintf(stderr,
                            "moe_trace: LOCKSTEP FAILURE doc %u trace layer %d stream %s: %lld "
                            "rows for %d tokens (model layer %d, node not emitted or spec wrong)\n",
                            doc.doc_id, l, names[s], (long long) got[s], n,
                            st.spec.layer_index_map[(size_t) l]);
                    rc = 7;
                    break;
                }
            }
            if (rc) break;
        }
        if (rc) break;

        if (opt.writes_trace() && !writer.end_document(doc.doc_id, token_ids.data(), doc_base)) {
            fprintf(stderr, "moe_trace: %s\n", writer.error.c_str());
            rc = 8;
            break;
        }
        n_docs_done++;
    }

    llama_batch_free(batch);

    const bool verified = opt.writes_trace() ? writer.close_and_verify() : true;
    if (rc == 0 && !verified) {
        fprintf(stderr, "moe_trace: %s\n", writer.error.c_str());
        rc = 9;
    }

    // Truncation is recorded in the stats file either way, but it also warrants a line in the
    // session log: it means the corpus cap was computed under a tokenizer that fragments this text
    // less than this model's does, so the corpus -- not the capture -- is what needs fixing.
    if (any_truncated) {
        fprintf(stderr,
                "moe_trace: WARNING %llu of %zu documents exceeded n_ctx=%d and were truncated "
                "(%llu tokens dropped, first doc_id %u). This model saw less text than a "
                "smaller-vocab model would for the same corpus.\n",
                (unsigned long long) n_docs_truncated, corpus.size(), opt.n_ctx,
                (unsigned long long) n_tokens_dropped, first_truncated_doc);
    }

    // Stats go to a file, not to stdout, so the runner parses data rather than scraping logs.
    if (FILE * sf = fopen(opt.stats_path.c_str(), "wb")) {
        fprintf(sf,
                "{\n"
                "  \"shard_id\": %d,\n"
                "  \"model_spec\": \"%s\",\n"
                "  \"n_docs\": %llu,\n"
                "  \"n_docs_in_shard\": %zu,\n"
                "  \"n_tokens\": %llu,\n"
                "  \"n_captured\": %llu,\n"
                "  \"n_docs_truncated\": %llu,\n"
                "  \"n_tokens_dropped\": %llu,\n"
                "  \"first_truncated_doc\": %lld,\n"
                "  \"n_moe_layers\": %d,\n"
                "  \"n_experts\": %d,\n"
                "  \"top_k\": %d,\n"
                "  \"hidden_dim\": %d,\n"
                "  \"hidden_stride\": %d,\n"
                "  \"index_scheme\": \"doc_id*n_ctx+pos_in_doc\",\n"
                "  \"index_doc_span\": %d,\n"
                "  \"nodes_captured\": %lld,\n"
                "  \"topk_layout\": \"%s\",\n"
                "  \"node_topk\": \"%s\",\n"
                "  \"node_logits\": \"%s\",\n"
                "  \"node_router_input\": \"%s\",\n"
                "  \"capture_mode\": \"%s\",\n"
                "  \"decode_seconds\": %.6f,\n"
                "  \"n_tokens_decoded\": %llu,\n"
                "  \"prefill_tok_per_s\": %.3f,\n"
                "  \"exit_code\": %d\n"
                "}\n",
                opt.shard_id, st.spec.model.c_str(), (unsigned long long) n_docs_done, corpus.size(),
                (unsigned long long) (opt.writes_trace() ? writer.total_tokens : n_tokens_decoded),
                (unsigned long long) (opt.writes_trace() ? writer.total_captured : 0),
                (unsigned long long) n_docs_truncated, (unsigned long long) n_tokens_dropped,
                any_truncated ? (long long) first_truncated_doc : -1LL,
                st.spec.n_moe_layers, st.spec.n_experts, st.spec.top_k, st.spec.hidden_dim,
                opt.hidden_stride, opt.n_ctx,
                (long long) st.n_nodes_captured,
                st.saw_strided_topk ? (st.saw_contiguous_topk ? "mixed" : "strided_view")
                                    : (st.saw_contiguous_topk ? "contiguous" : "none"),
                st.spec.tpl_topk.c_str(), st.spec.tpl_logits.c_str(),
                st.spec.tpl_router_input.c_str(),
                opt.mode == capture_mode::full ? "full"
                    : (opt.mode == capture_mode::filter_off ? "filter-off" : "no-callback"),
                decode_seconds, (unsigned long long) n_tokens_decoded,
                decode_seconds > 0.0 ? (double) n_tokens_decoded / decode_seconds : 0.0,
                rc);
        fclose(sf);
    }

    // A "mixed" layout means some ubatches came back packed and some strided. Nothing in ggml
    // should do that, and if it happens the de-striding logic is reading a tensor whose layout it
    // has misidentified -- so it is a hard failure, not a warning.
    if (rc == 0 && st.saw_strided_topk && st.saw_contiguous_topk) {
        fprintf(stderr, "moe_trace: topk arrived both packed and strided; layout is not stable\n");
        rc = 10;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
