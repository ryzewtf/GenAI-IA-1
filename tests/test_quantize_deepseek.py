"""DeepSeek W4A16 self-quantization — the router-skip rule (CPU, no GPTQModel import).

The one thing that must be right before any GPU quantization run: the dynamic negative-match keeps
the MoE router (.mlp.gate) in fp16 and nothing else. A W4 router would silently degrade expert
selection — the study's measured object. These assert the pure regex, so they run without GPTQModel.
"""

from __future__ import annotations

from scripts.quantize_deepseek import (
    ROUTER_SKIP_PATTERN,
    dynamic_skip_config,
    router_gate_is_skipped,
)


def test_skip_matches_every_layer_router_gate():
    for name in ("model.layers.1.mlp.gate", "model.layers.9.mlp.gate", "model.layers.26.mlp.gate"):
        assert router_gate_is_skipped(name), name


def test_skip_does_not_match_experts_shared_or_dense_ffn():
    # routed experts, shared experts, and the dense layer-0 FFN must all be QUANTIZED (not skipped).
    for name in (
        "model.layers.1.mlp.experts.0.down_proj",
        "model.layers.1.mlp.shared_experts.gate_up_proj",
        "model.layers.0.mlp.gate_up_proj",   # dense layer 0 starts with .mlp.gate but is not the router
        "model.layers.1.mlp.gate_up_proj",
        "model.layers.1.self_attn.q_proj",
    ):
        assert not router_gate_is_skipped(name), name


def test_dynamic_config_is_the_negative_match():
    cfg = dynamic_skip_config()
    assert cfg == {ROUTER_SKIP_PATTERN: {}}
    assert ROUTER_SKIP_PATTERN.startswith("-:")  # GPTQModel's exclude prefix
