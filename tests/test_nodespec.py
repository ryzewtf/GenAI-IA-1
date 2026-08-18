"""Node spec tests — plan T1.4 / T2.1.

The C++ harness cannot be compiled here (W.5 is blocked on a build generator), so the rules the
spec has to satisfy are implemented in Python as well and tested there. ``resolve_lookup`` mirrors
the map construction in ``node_spec.hpp`` line for line, which is what makes the collision and
layer-map checks testable today instead of at first collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.capture.nodespec import (
    DEFAULT_NODE_TEMPLATES,
    SELECTION_CHAIN,
    NodeSpec,
    NodeSpecError,
    build_spec,
    layer_index_map,
    main,
    parse_spec,
    resolve_lookup,
    selection_chain,
)

MODELS_YAML = Path("configs/models.yaml")


@pytest.fixture(scope="module")
def models():
    return yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))


def _spec(**overrides) -> NodeSpec:
    base = dict(
        model="synth",
        n_moe_layers=3,
        n_experts=8,
        top_k=2,
        hidden_dim=4,
        layer_map=[0, 1, 2],
        node_topk="ffn_moe_topk-%d",
        node_logits="ffn_moe_probs-%d",
        node_router_input="ffn_norm-%d",
    )
    base.update(overrides)
    return NodeSpec(**base)


# -- layer mapping ---------------------------------------------------------------------------------


def test_layer_map_is_identity_when_every_layer_is_moe():
    assert layer_index_map({"n_moe_layers": 4, "moe_layer_offset": 0}) == [0, 1, 2, 3]


def test_deepseek_layer_map_is_offset_by_the_dense_first_layer(models):
    """T3.5: first_k_dense_replace=1 means trace layer 0 is model layer 1, for 26 layers."""
    config = models["models"]["deepseek-v2-lite"]
    mapping = layer_index_map(config)
    assert mapping == list(range(1, 27))
    assert len(mapping) == config["n_moe_layers"] == 26


def test_interleaved_moe_layers_are_refused_rather_than_mismapped():
    """A contiguous map would be silently wrong: right file sizes, every layer mislabelled."""
    with pytest.raises(NodeSpecError, match="not contiguous"):
        layer_index_map({"n_moe_layers": 4, "moe_layer_offset": 1, "moe_layer_freq": 2})


@pytest.mark.parametrize("bad", [{"n_moe_layers": 0}, {"n_moe_layers": 4, "moe_layer_offset": -1}])
def test_layer_map_rejects_nonsense(bad):
    with pytest.raises(NodeSpecError):
        layer_index_map(bad)


# -- the selection chain ----------------------------------------------------------------------------


def test_plain_model_selects_on_probs():
    present, selection = selection_chain({})
    assert present == ["ffn_moe_logits", "ffn_moe_probs"]
    assert selection == "ffn_moe_probs"


def test_router_bias_inserts_the_biased_logits_node(models):
    """§1.6: GPT-OSS supplies gate_inp_b, so the raw logits are not what selection saw."""
    config = models["models"]["gpt-oss-20b"]
    assert config["has_router_bias"] is True
    present, selection = selection_chain(config)
    assert "ffn_moe_logits_biased" in present
    assert present.index("ffn_moe_logits_biased") < present.index("ffn_moe_probs")
    assert selection == "ffn_moe_probs"


def test_expert_selection_bias_and_group_mask_move_the_selection_node():
    _, selection = selection_chain({"has_expert_selection_bias": True})
    assert selection == "ffn_moe_probs_biased"
    _, selection = selection_chain({"has_expert_selection_bias": True, "n_expert_groups": 8})
    assert selection == "ffn_moe_probs_masked"


def test_deepseek_v2_lite_has_neither(models):
    """V2-Lite is plain greedy top-k: exp_probs_b is V3, and n_group is 1."""
    config = models["models"]["deepseek-v2-lite"]
    _, selection = selection_chain(config)
    assert selection == "ffn_moe_probs"


def test_selection_chain_is_a_prefix_of_the_declared_chain():
    for config in ({}, {"has_router_bias": True}, {"has_expert_selection_bias": True, "n_expert_groups": 4}):
        present, _ = selection_chain(config)
        assert all(node in SELECTION_CHAIN for node in present)
        positions = [SELECTION_CHAIN.index(n) for n in present]
        assert positions == sorted(positions)  # chain order, not arbitrary order


# -- lookup construction (mirrors node_spec.hpp) ------------------------------------------------------


def test_lookup_covers_every_stream_and_layer():
    lookup = resolve_lookup(_spec())
    assert len(lookup) == 3 * 3
    assert lookup["ffn_moe_topk-2"] == ("topk", 2, 2)
    assert lookup["ffn_norm-0"] == ("router_input", 0, 0)


def test_lookup_uses_the_model_layer_in_the_name_not_the_trace_layer():
    lookup = resolve_lookup(_spec(n_moe_layers=2, layer_map=[1, 2]))
    assert lookup["ffn_moe_topk-1"] == ("topk", 0, 1)
    assert "ffn_moe_topk-0" not in lookup


def test_two_streams_resolving_to_one_tensor_is_an_error():
    """Would double-write one stream and leave another empty — files of the right size, no data."""
    with pytest.raises(NodeSpecError, match="both resolve to tensor"):
        resolve_lookup(_spec(node_logits="ffn_moe_topk-%d"))


def test_duplicate_model_layers_are_rejected():
    with pytest.raises(NodeSpecError, match="duplicate model layers"):
        resolve_lookup(_spec(layer_map=[0, 1, 1]))


def test_top_k_above_n_experts_is_rejected():
    with pytest.raises(NodeSpecError, match="exceeds n_experts"):
        resolve_lookup(_spec(top_k=9))


def test_layer_map_length_must_match_n_moe_layers():
    with pytest.raises(NodeSpecError, match="layer_index_map has"):
        resolve_lookup(_spec(layer_map=[0, 1]))


# -- build_spec: T1.4 as a halt gate --------------------------------------------------------------------


def test_unverified_node_names_are_refused_by_default(models):
    """T1.4 is a HALT GATE. Guessing node names is exactly what it exists to prevent."""
    with pytest.raises(NodeSpecError, match="HALT GATE"):
        build_spec("olmoe-0125", models)


def test_assume_defaults_produces_an_explicitly_unverified_spec(models):
    spec = build_spec("olmoe-0125", models, assume_defaults=True)
    assert spec.verified is False
    assert spec.node_topk == DEFAULT_NODE_TEMPLATES["topk"]
    assert spec.node_router_input == DEFAULT_NODE_TEMPLATES["router_input"]
    assert "verified_by_T1.4=false" in spec.render()
    assert any("UNVERIFIED" in n for n in spec.notes)


def test_hypothesis_spec_uses_the_predicted_selection_node(models):
    spec = build_spec("gpt-oss-20b", models, assume_defaults=True)
    _, selection = selection_chain(models["models"]["gpt-oss-20b"])
    assert spec.node_logits == f"{selection}-%d"


def test_filled_node_names_without_logit_tensor_used_is_refused(models):
    config = dict(models["models"]["olmoe-0125"])
    config["node_names"] = {"topk": "ffn_moe_topk-%d", "router_input": "ffn_norm-%d"}
    config["logit_tensor_used"] = None
    patched = {"models": {"olmoe-0125": config}}
    with pytest.raises(NodeSpecError, match="logit_tensor_used"):
        build_spec("olmoe-0125", patched)


def test_a_verified_spec_round_trips(models):
    config = dict(models["models"]["deepseek-v2-lite"])
    config["node_names"] = {"topk": "ffn_moe_topk-%d", "router_input": "ffn_norm-%d"}
    config["logit_tensor_used"] = "ffn_moe_probs"
    spec = build_spec("deepseek-v2-lite", {"models": {"deepseek-v2-lite": config}})
    assert spec.verified is True

    back = parse_spec(spec.render())
    assert back.layer_map == spec.layer_map == list(range(1, 27))
    assert back.n_experts == 64 and back.top_k == 6
    assert back.node_logits == "ffn_moe_probs-%d"


def test_templates_must_carry_exactly_one_placeholder(models):
    config = dict(models["models"]["olmoe-0125"])
    config["node_names"] = {"topk": "ffn_moe_topk", "router_input": "ffn_norm-%d"}
    config["logit_tensor_used"] = "ffn_moe_probs"
    with pytest.raises(NodeSpecError, match="placeholder"):
        build_spec("olmoe-0125", {"models": {"olmoe-0125": config}})


def test_logits_template_disagreeing_with_logit_tensor_used_is_refused(models):
    """The stream and the record must name the same node.

    T1.4 writes both: `node_names.logits` is what the capture callback matches on, and
    `logit_tensor_used` is what T8.2 and the paper quote as the tensor the margins came from.
    A hand-edit that changes one and not the other produces a trace whose margins are real but
    attributed to the wrong node, and nothing downstream can see it.
    """
    config = dict(models["models"]["olmoe-0125"])
    config["node_names"] = {"topk": "ffn_moe_topk-%d", "router_input": "ffn_norm-%d",
                            "logits": "ffn_moe_logits-%d"}
    config["logit_tensor_used"] = "ffn_moe_probs"
    with pytest.raises(NodeSpecError, match="does not name"):
        build_spec("olmoe-0125", {"models": {"olmoe-0125": config}})


def test_logit_tensor_used_alone_still_yields_a_template(models):
    """node_names.logits may be absent; the bare record is enough to build the template."""
    config = dict(models["models"]["olmoe-0125"])
    config["node_names"] = {"topk": "ffn_moe_topk-%d", "router_input": "ffn_norm-%d"}
    config["logit_tensor_used"] = "ffn_moe_probs"
    spec = build_spec("olmoe-0125", {"models": {"olmoe-0125": config}})
    assert spec.node_logits == "ffn_moe_probs-%d"


def test_rendered_spec_is_pure_ascii_and_parses(models):
    spec = build_spec("qwen3-30b-a3b", models, assume_defaults=True)
    text = spec.render()
    text.encode("ascii")  # the C++ parser is byte-oriented; no UTF-8 surprises
    assert parse_spec(text).model == "qwen3-30b-a3b"


def test_parse_spec_rejects_a_wrong_format_version():
    with pytest.raises(NodeSpecError, match="format_version"):
        parse_spec("format_version=2\nmodel=x\n")


def test_parse_spec_names_the_missing_keys():
    with pytest.raises(NodeSpecError, match="missing key"):
        parse_spec("format_version=1\nmodel=x\n")


# -- every model in the panel --------------------------------------------------------------------------


def test_every_panel_model_produces_a_valid_hypothesis_spec(models):
    """One unbuildable spec would surface at collection time; this surfaces it now."""
    for key in models["models"]:
        spec = build_spec(key, models, assume_defaults=True)
        assert len(resolve_lookup(spec)) == 3 * spec.n_moe_layers
        assert spec.layer_map[0] == int(models["models"][key].get("moe_layer_offset", 0) or 0)
        parse_spec(spec.render())


def test_cli_writes_one_spec_per_model(tmp_path, capsys):
    rc = main(
        [
            "--models",
            str(MODELS_YAML),
            "--model",
            "all",
            "--out-dir",
            str(tmp_path),
            "--assume-defaults",
        ]
    )
    assert rc == 0
    written = sorted(p.stem for p in tmp_path.glob("*.spec"))
    assert written == sorted(yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"])
    assert "UNVERIFIED" in capsys.readouterr().out
