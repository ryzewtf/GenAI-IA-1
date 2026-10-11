"""The compressed-tensors config patcher — makes a ct-0.16-written config loadable by ct-0.11 (vLLM 0.10.2).

These are torch-free: they assert the pure dict surgery that flushes the producer/consumer skew class
(see scripts/patch_quant_config docstring). Two real symptoms burned a paid GPU session each —
scale_dtype/zp_dtype and an illegal per-channel actorder — so each gets a direct test, plus the
general allowlist-flush and the two hard-refusal guards that keep a third symptom off the GPU.
"""

from __future__ import annotations

import copy

import pytest

from scripts.patch_quant_config import (
    CT_0_11_ARG_FIELDS,
    find_gidx_tensors,
    flush_nonallowlisted_arg_keys,
    patch_config_dict,
    strip_illegal_actorder,
    strip_incompatible_keys,
)


def _w8a8_config() -> dict:
    """A realistic DeepSeek W8A8-int8 config.json as compressed-tensors 0.16 writes it: per-channel
    weights with an (illegal-for-channel) actorder, plus the null scale_dtype/zp_dtype cruft."""
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "scale_dtype": None,
            "zp_dtype": None,
            "ignore": ["lm_head", "re:.*\\.mlp\\.gate$"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8, "type": "int", "symmetric": True,
                        "strategy": "channel", "group_size": None,
                        "actorder": "static",          # illegal with channel strategy on ct 0.11
                        "scale_dtype": None, "zp_dtype": None,
                    },
                    "input_activations": {
                        "num_bits": 8, "type": "int", "symmetric": True,
                        "strategy": "token", "dynamic": True,
                        "scale_dtype": None, "zp_dtype": None,
                    },
                },
            },
        },
    }


def test_strip_incompatible_keys_removes_null_dtypes_everywhere():
    cfg = _w8a8_config()
    removed = strip_incompatible_keys(cfg)
    # top-level + both inner-arg copies of each key
    assert set(removed) == {
        "quantization_config.scale_dtype",
        "quantization_config.zp_dtype",
        "quantization_config.config_groups.group_0.weights.scale_dtype",
        "quantization_config.config_groups.group_0.weights.zp_dtype",
        "quantization_config.config_groups.group_0.input_activations.scale_dtype",
        "quantization_config.config_groups.group_0.input_activations.zp_dtype",
    }
    qc = cfg["quantization_config"]
    assert "scale_dtype" not in qc and "zp_dtype" not in qc
    assert "scale_dtype" not in qc["config_groups"]["group_0"]["weights"]


def test_strip_illegal_actorder_nulls_channel_actorder_only():
    cfg = _w8a8_config()
    removed = strip_illegal_actorder(cfg)
    assert removed == ["quantization_config.config_groups.group_0.weights.actorder"]
    assert "actorder" not in cfg["quantization_config"]["config_groups"]["group_0"]["weights"]


def test_strip_illegal_actorder_keeps_actorder_on_group_strategy():
    cfg = _w8a8_config()
    cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] = "group"
    cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 128
    assert strip_illegal_actorder(cfg) == []  # legal with group strategy — left intact
    assert cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["actorder"] == "static"


def test_flush_drops_nonallowlisted_null_keys_but_not_real_fields():
    cfg = _w8a8_config()
    removed = flush_nonallowlisted_arg_keys(cfg)
    # only scale_dtype/zp_dtype are outside the ct-0.11 allowlist here (actorder IS allowlisted)
    assert all(d.endswith((".scale_dtype", ".zp_dtype")) for d in removed), removed
    w = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    for keep in ("num_bits", "type", "symmetric", "strategy", "actorder"):
        assert keep in w


def test_flush_refuses_to_drop_a_nonnull_unknown_field():
    """A new ct-0.16 field with a REAL value must stop the patch, not be silently discarded —
    that is exactly the 'surfaces only on paid GPU' failure we are trying to kill on CPU."""
    cfg = _w8a8_config()
    cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["mystery_scale"] = 0.125
    with pytest.raises(SystemExit, match="not a compressed-tensors 0.11.0 QuantizationArgs field"):
        flush_nonallowlisted_arg_keys(cfg)


def test_patch_config_dict_produces_a_ct_0_11_clean_config():
    cfg = _w8a8_config()
    patch_config_dict(cfg)
    w = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    ia = cfg["quantization_config"]["config_groups"]["group_0"]["input_activations"]
    # every surviving inner-args key is in the consumer allowlist; actorder gone; dtypes gone
    assert set(w) <= CT_0_11_ARG_FIELDS and "actorder" not in w
    assert set(ia) <= CT_0_11_ARG_FIELDS
    assert "scale_dtype" not in cfg["quantization_config"]


def test_patch_config_dict_is_idempotent():
    cfg = _w8a8_config()
    patch_config_dict(cfg)
    once = copy.deepcopy(cfg)
    assert patch_config_dict(cfg) == []     # second pass changes nothing
    assert cfg == once


def test_patch_config_dict_noop_on_already_clean_config():
    clean = {"quantization_config": {"config_groups": {"group_0": {"weights": {
        "num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"}}}}}
    before = copy.deepcopy(clean)
    assert patch_config_dict(clean) == []
    assert clean == before


# ---- g_idx orphan guard: nulling actorder is only safe when no permutation tensors exist ----

def test_find_gidx_tensors_detects_permutation_weights():
    keys = ["model.layers.1.mlp.experts.0.down_proj.weight_g_idx",
            "model.layers.1.mlp.experts.0.down_proj.weight_scale",
            "model.layers.1.mlp.gate.weight"]
    assert find_gidx_tensors(keys) == ["model.layers.1.mlp.experts.0.down_proj.weight_g_idx"]
    assert find_gidx_tensors(["model.layers.1.mlp.gate.weight"]) == []
    assert find_gidx_tensors(None) == []


def test_patch_refuses_when_actorder_strip_would_orphan_a_permutation():
    cfg = _w8a8_config()  # has a channel actorder to strip
    gidx_keys = ["model.layers.1.mlp.experts.0.down_proj.weight_g_idx"]
    with pytest.raises(SystemExit, match="orphan|permutation|column-permuted"):
        patch_config_dict(cfg, weight_keys=gidx_keys)


def test_patch_allows_strip_when_no_permutation_tensors_present():
    cfg = _w8a8_config()
    # channel-strategy GPTQ writes no g_idx; the normal expert tensors must not trip the guard
    ok_keys = ["model.layers.1.mlp.experts.0.down_proj.weight_scale",
               "model.layers.1.mlp.experts.0.down_proj.weight"]
    removed = patch_config_dict(cfg, weight_keys=ok_keys)
    assert any(d.endswith(".actorder") for d in removed)
