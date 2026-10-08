"""DeepSeek W4A16 self-quantization — the router-ignore rule (CPU, no llm-compressor import).

The one thing that must be right before any GPU quantization run: the GPTQModifier ``ignore`` keeps
the MoE router (.mlp.gate) in fp16 and nothing else. A W4 router would silently degrade expert
selection — the study's measured object. These assert the pure regex, so they run without
llm-compressor (which is a GPU-env-only dependency).
"""

from __future__ import annotations

from scripts.quantize_deepseek import (
    IGNORE_PATTERNS,
    ROUTER_IGNORE_REGEX,
    router_gate_is_ignored,
)


def test_ignore_matches_every_layer_router_gate():
    for name in ("model.layers.1.mlp.gate", "model.layers.9.mlp.gate", "model.layers.26.mlp.gate"):
        assert router_gate_is_ignored(name), name


def test_ignore_does_not_match_experts_shared_or_dense_ffn():
    # routed experts, shared experts, and the dense layer-0 FFN must all be QUANTIZED (not ignored).
    for name in (
        "model.layers.1.mlp.experts.0.down_proj",
        "model.layers.1.mlp.shared_experts.gate_proj",
        "model.layers.1.mlp.shared_experts.gate_up_proj",
        "model.layers.0.mlp.gate_proj",      # dense layer 0 starts with .mlp.gate but is not the router
        "model.layers.1.mlp.gate_up_proj",
        "model.layers.1.self_attn.q_proj",
    ):
        assert not router_gate_is_ignored(name), name


def test_ignore_list_has_lm_head_and_router_regex():
    assert "lm_head" in IGNORE_PATTERNS
    assert ROUTER_IGNORE_REGEX in IGNORE_PATTERNS
    assert ROUTER_IGNORE_REGEX.startswith("re:")  # llm-compressor's regex-matcher prefix
