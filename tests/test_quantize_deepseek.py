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
    assess_quant,
    module_is_ignored,
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


def test_full_ignore_keeps_router_lmhead_and_dense_layer0_fp16():
    # fp16 (ignored): router gate, lm_head, and the dense layer-0 FFN (10944 cols, indivisible by 128).
    for name in (
        "model.layers.1.mlp.gate",
        "model.layers.26.mlp.gate",
        "lm_head",
        "model.lm_head",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ):
        assert module_is_ignored(name), name


def test_full_ignore_quantizes_moe_experts_and_other_dense_layers():
    # QUANTIZED (not ignored): routed/shared experts, and FFN projections in MoE layers (>=1).
    for name in (
        "model.layers.1.mlp.experts.0.down_proj",
        "model.layers.5.mlp.experts.63.gate_proj",
        "model.layers.1.mlp.shared_experts.down_proj",
        "model.layers.2.mlp.down_proj",      # a MoE layer's own down_proj is NOT the dense layer 0
        "model.layers.1.self_attn.q_proj",
    ):
        assert not module_is_ignored(name), name


# ---- assess_quant: the pre-upload verdict (pure; no GPU / no saved model needed) ----

_GOOD_KEYS = [
    "model.layers.1.mlp.gate.weight",                      # router: plain fp16 weight, NOT packed
    "model.layers.1.mlp.experts.0.down_proj.weight_packed",
    "model.layers.1.mlp.experts.0.down_proj.weight_scale",
    "model.layers.1.mlp.shared_experts.up_proj.weight_packed",
    "lm_head.weight",
]
_CFG_GROUPS = {"group_0": {"weights": {"num_bits": 4}}}
_IGNORE = ["lm_head", "re:.*\\.mlp\\.gate$", "re:model\\.layers\\.0\\.mlp\\.\\w+_proj$"]


def test_assess_quant_accepts_router_fp16_experts_packed():
    ok, msg = assess_quant(_GOOD_KEYS, ignore=_IGNORE, config_groups=_CFG_GROUPS)
    assert ok, msg


def test_assess_quant_hard_fails_on_quantized_router():
    bad = _GOOD_KEYS + ["model.layers.1.mlp.gate.weight_packed"]
    ok, msg = assess_quant(bad, ignore=_IGNORE, config_groups=_CFG_GROUPS)
    assert not ok and "router" in msg.lower()


def test_assess_quant_hard_fails_when_nothing_compressed():
    plain = ["model.layers.1.mlp.gate.weight", "model.layers.1.mlp.experts.0.down_proj.weight"]
    ok, msg = assess_quant(plain, ignore=_IGNORE, config_groups=None)
    assert not ok and "nothing was quantized" in msg.lower()


def test_assess_quant_does_not_false_fail_on_surprising_expert_key_names():
    # config_groups present + no gate packed → treat as OK even if expert key spelling is unexpected,
    # so a good ~30-min run is never thrown away over a key-naming quirk.
    surprising = ["model.layers.1.mlp.gate.weight", "some.other.module.weight_packed"]
    ok, msg = assess_quant(surprising, ignore=_IGNORE, config_groups=_CFG_GROUPS)
    assert ok, msg
