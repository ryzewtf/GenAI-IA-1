// Node spec: which ggml graph nodes to capture, resolved per model layer.
//
// Plan T1.4 discovers the node names per model and is a HALT GATE that must pass before a byte of
// trace is written, because the discovered names determine the stream layout. Nothing here is
// hardcoded for that reason: the spec is a text file generated from configs/models.yaml by
// src/capture/nodespec.py, so T1.4's output is data, not a code change.
//
// Verified against llama.cpp master, llm_graph_context::build_moe_ffn (src/llama-graph.cpp):
//
//   logits = build_lora_mm(gate_inp, cur)          -> cb "ffn_moe_logits"        [n_expert, n_tok]
//   if (gate_inp_b)  logits += gate_inp_b          -> cb "ffn_moe_logits_biased"     <- GPT-OSS
//   probs  = gating_op(logits)                     -> cb "ffn_moe_probs"
//   if (exp_probs_b) selection_probs = probs + b   -> cb "ffn_moe_probs_biased"       <- DeepSeek-V3
//   if (n_expert_groups > 1) group mask            -> cb "ffn_moe_probs_masked"
//   selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)
//                                                  -> cb "ffn_moe_argsort" on the PARENT
//                                                  -> cb "ffn_moe_topk"    on the VIEW    [I32]
//
// Two consequences that the spec exists to encode:
//
//   1. Top-k does NOT run on `ffn_moe_logits`. It runs on `selection_probs`, the last node in that
//      chain. For softmax/sigmoid/sqrt-softplus gating the transform is monotonic so the ORDER is
//      unchanged, but `exp_probs_b` and group masking are NOT order-preserving. So `node_logits`
//      must name whichever node selection actually consumed, and T1.4 verifies by recomputing
//      top-k from it and comparing against the captured topk.bin.
//   2. The router input is the tensor passed as `cur`, which most architectures cb as
//      "ffn_norm-<il>" one line earlier. It is the *normalized* hidden state, which is exactly what
//      plan §1.3/C10 requires F4/F5/FV to condition on. It varies by architecture, so it is a
//      spec field and the FV probe (T3.3, >=0.99) is its end-to-end validation.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace moe {

enum class stream_kind : int {
    none = 0,
    topk = 1,
    logits = 2,
    router_input = 3,
};

struct node_target {
    stream_kind kind = stream_kind::none;
    int trace_layer = -1;  // index into the trace's layer axis, 0..n_moe_layers-1
    int model_layer = -1;  // llama.cpp's `il`
};

struct node_spec {
    std::string model;
    int n_moe_layers = 0;
    int n_experts = 0;
    int top_k = 0;
    int hidden_dim = 0;

    std::string tpl_topk;
    std::string tpl_logits;
    std::string tpl_router_input;

    // trace layer -> model layer. DeepSeek-V2-Lite has first_k_dense_replace=1, so trace layer 0
    // is model layer 1 and a naive identity map silently mislabels every layer (plan T3.5).
    std::vector<int> layer_index_map;

    std::unordered_map<std::string, node_target> lookup;
    std::string error;

    bool ok() const { return error.empty(); }

    const node_target * find(const char * name) const {
        auto it = lookup.find(name);
        return it == lookup.end() ? nullptr : &it->second;
    }
};

// Substitute the single "%d" in a template. Deliberately NOT snprintf: the template comes from a
// config file, and handing an untrusted string to a format function is a bug waiting for a config
// typo. This also fails loudly on a template with no placeholder.
inline bool expand_template(const std::string & tpl, int value, std::string & out) {
    const size_t at = tpl.find("%d");
    if (at == std::string::npos) {
        return false;
    }
    if (tpl.find("%d", at + 2) != std::string::npos) {
        return false;  // more than one placeholder is ambiguous
    }
    out = tpl.substr(0, at) + std::to_string(value) + tpl.substr(at + 2);
    return true;
}

inline std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline std::vector<int> parse_int_list(const std::string & s, bool & ok) {
    std::vector<int> out;
    ok = true;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string piece = trim(s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (!piece.empty()) {
            char * end = nullptr;
            const long v = std::strtol(piece.c_str(), &end, 10);
            if (end == piece.c_str() || *end != '\0' || v < 0) {
                ok = false;
                return out;
            }
            out.push_back((int) v);
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

// Parse a `key=value` spec file and build the name -> target map.
inline node_spec load_node_spec(const char * path) {
    node_spec spec;

    FILE * f = fopen(path, "rb");
    if (!f) {
        spec.error = std::string("cannot open node spec: ") + path;
        return spec;
    }

    std::unordered_map<std::string, std::string> kv;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        std::string s = trim(line);
        if (s.empty() || s[0] == '#') continue;
        const size_t eq = s.find('=');
        if (eq == std::string::npos) {
            spec.error = "malformed spec line (no '='): " + s;
            fclose(f);
            return spec;
        }
        kv[trim(s.substr(0, eq))] = trim(s.substr(eq + 1));
    }
    fclose(f);

    auto need = [&](const char * key, std::string & dst) {
        auto it = kv.find(key);
        if (it == kv.end() || it->second.empty()) {
            if (spec.error.empty()) spec.error = std::string("spec is missing key: ") + key;
            return false;
        }
        dst = it->second;
        return true;
    };
    auto need_int = [&](const char * key, int & dst) {
        std::string raw;
        if (!need(key, raw)) return false;
        char * end = nullptr;
        const long v = std::strtol(raw.c_str(), &end, 10);
        if (end == raw.c_str() || *end != '\0' || v <= 0) {
            if (spec.error.empty()) spec.error = std::string("spec key must be a positive integer: ") + key;
            return false;
        }
        dst = (int) v;
        return true;
    };

    std::string version_raw;
    if (need("format_version", version_raw) && version_raw != "1") {
        spec.error = "unsupported spec format_version: " + version_raw;
        return spec;
    }

    need("model", spec.model);
    need_int("n_moe_layers", spec.n_moe_layers);
    need_int("n_experts", spec.n_experts);
    need_int("top_k", spec.top_k);
    need_int("hidden_dim", spec.hidden_dim);
    need("node_topk", spec.tpl_topk);
    need("node_logits", spec.tpl_logits);
    need("node_router_input", spec.tpl_router_input);
    if (!spec.ok()) return spec;

    std::string map_raw;
    if (!need("layer_index_map", map_raw)) return spec;
    bool list_ok = false;
    spec.layer_index_map = parse_int_list(map_raw, list_ok);
    if (!list_ok) {
        spec.error = "layer_index_map is not a comma-separated list of non-negative integers";
        return spec;
    }
    if ((int) spec.layer_index_map.size() != spec.n_moe_layers) {
        spec.error = "layer_index_map has " + std::to_string(spec.layer_index_map.size()) +
                     " entries but n_moe_layers=" + std::to_string(spec.n_moe_layers);
        return spec;
    }
    if (spec.top_k > spec.n_experts) {
        spec.error = "top_k exceeds n_experts";
        return spec;
    }

    struct entry { const std::string * tpl; stream_kind kind; };
    const entry entries[] = {
        { &spec.tpl_topk,         stream_kind::topk         },
        { &spec.tpl_logits,       stream_kind::logits       },
        { &spec.tpl_router_input, stream_kind::router_input },
    };

    for (int trace_layer = 0; trace_layer < spec.n_moe_layers; ++trace_layer) {
        const int model_layer = spec.layer_index_map[trace_layer];
        for (const entry & e : entries) {
            std::string name;
            if (!expand_template(*e.tpl, model_layer, name)) {
                spec.error = "node template needs exactly one \"%d\" placeholder: " + *e.tpl;
                return spec;
            }
            // A collision means two streams resolve to the same tensor, which would double-write
            // one stream and leave another empty. It is a spec bug, not something to tolerate.
            if (spec.lookup.count(name)) {
                spec.error = "two streams resolve to the same node name: " + name;
                return spec;
            }
            spec.lookup[name] = node_target{ e.kind, trace_layer, model_layer };
        }
    }

    return spec;
}

}  // namespace moe
