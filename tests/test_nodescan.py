"""Tests for T1.4, the node-name halt gate (`src/capture/nodescan.py`).

Every test synthesizes an `llama-eval-callback` dump in the exact format of
`common/debug.cpp:172` at the pinned commit. Nothing here loads a model or spawns a subprocess.

The gate's job is to catch the *duplicate* node, which is the failure that produces a trace of
exactly the right size with unreconstructable row order. So most of these tests construct a
graph that is wrong in a way a name-only check would accept.
"""

from __future__ import annotations

import pytest

from src.capture.nodescan import (
    NodeScanError,
    confirm_nodes,
    inventory_by_layer,
    main,
    parse_eval_callback,
)
from src.capture.nodespec import NodeSpecError

# A stand-in for `olmoe`: 4 MoE layers from il=0, 64 experts, top-8, hidden 2048.
CFG = {
    "key": "olmoe-test",
    "arch": "olmoe",
    "n_moe_layers": 4,
    "moe_layer_offset": 0,
    "n_experts": 64,
    "top_k": 8,
    "hidden_dim": 2048,
    "has_router_bias": False,
    "has_expert_selection_bias": False,
    "n_expert_groups": 1,
    "post_topk": "none",
}


def _line(name: str, dtype: str, op: str, ne: tuple[int, ...], src0: str = "x",
          src0_ne: tuple[int, ...] = (1, 1, 1, 1), src1: str | None = None,
          src1_ne: tuple[int, ...] = (1, 1, 1, 1)) -> str:
    """One node line, matching common/debug.cpp's format string including its stray brace."""
    ne_s = ", ".join(str(v) for v in ne)
    s0 = ", ".join(str(v) for v in src0_ne)
    s1 = f"{src1}{{{', '.join(str(v) for v in src1_ne)}}}" if src1 else ""
    return (f"common_debug_cb_eval: {name:>24s} = ({dtype}) {op:>10s}"
            f"({src0}{{{s0}}}, {s1}}}) = {{{ne_s}}}")


def _dump(
    n_layers: int = 4,
    *,
    n_tokens: int = 5,
    n_experts: int = 64,
    top_k: int = 8,
    hidden: int = 2048,
    router_input: str = "ffn_norm",
    duplicate: str | None = None,
    omit: str | None = None,
    extra_chain: str | None = None,
    topk_dtype: str = "i32",
    logits_srcs: bool = True,
    router_input_of_layer=None,
) -> str:
    """A healthy graph dump, with one optional defect injected.

    `logits_srcs=False` drops the source list from the logits node, which is what forces the
    name-fallback path. `router_input_of_layer` is a callable il -> name, for building a graph
    whose router input differs between layers.
    """
    out = [
        "build: 7077abbe with cc (GCC) 15.1.0 for x86_64-w64-mingw32",
        "llama_model_loader: loaded meta data with 30 key-value pairs",
        _line("inp_embd", "f32", "get_rows", (hidden, n_tokens, 1, 1)),
    ]
    for il in range(n_layers):
        ri = router_input_of_layer(il) if router_input_of_layer else router_input
        out.append(_line(f"attn_norm-{il}", "f32", "mul", (hidden, n_tokens, 1, 1)))
        out.append(_line(f"{ri}-{il}", "f32", "mul", (hidden, n_tokens, 1, 1)))
        if omit != "ffn_moe_probs":
            # mul_mat(ffn_gate_inp, cur) -- the second operand names the router input, which is
            # how the scanner resolves it without guessing.
            out.append(_line(
                f"ffn_moe_logits-{il}", "f32", "mul_mat", (n_experts, n_tokens, 1, 1),
                # src0 is always the router weight. Dropping src1 is the realistic "no usable
                # source" case: the only name reported is one we must not select.
                src0=f"blk.{il}.ffn_gate_inp.weight",
                src0_ne=(hidden, n_experts, 1, 1),
                src1=f"{ri}-{il}" if logits_srcs else None,
                src1_ne=(hidden, n_tokens, 1, 1),
            ))
            out.append(_line(f"ffn_moe_probs-{il}", "f32", "soft_max", (n_experts, n_tokens, 1, 1),
                             src0=f"ffn_moe_logits-{il}", src0_ne=(n_experts, n_tokens, 1, 1)))
        if extra_chain:
            out.append(_line(f"{extra_chain}-{il}", "f32", "add", (n_experts, n_tokens, 1, 1)))
        if omit != "ffn_moe_topk":
            out.append(_line(f"ffn_moe_topk-{il}", topk_dtype, "view", (top_k, n_tokens, 1, 1)))
        if duplicate:
            shape = (top_k, n_tokens, 1, 1) if duplicate == "ffn_moe_topk" else (
                (hidden, n_tokens, 1, 1) if duplicate == router_input else (n_experts, n_tokens, 1, 1))
            out.append(_line(f"{duplicate}-{il}", "f32", "cont", shape))
        # Interleaved value dumps, exactly as the tool emits them.
        out.append("                                     [")
        out.append("                                      [ 0.1234, 0.5678, 0.9012, ...],")
        out.append("                                     ]")
        out.append("                                     sum = 1.000000")
    out.append(_line("result_output", "f32", "mul_mat", (50304, n_tokens, 1, 1)))
    return "\n".join(out)


# --- parsing ---------------------------------------------------------------------------------


def test_node_lines_are_parsed_and_the_interleaved_value_dumps_are_not():
    obs = parse_eval_callback(_dump(n_layers=2))
    names = [o.name for o in obs]
    assert "ffn_moe_topk-0" in names and "ffn_moe_topk-1" in names
    # 2 layers x (attn_norm, ffn_norm, logits, probs, topk) + inp_embd + result_output
    assert len(obs) == 2 * 5 + 2
    assert all(not o.name.startswith("[") for o in obs), "a value-dump line was parsed as a node"
    assert all(not o.name.startswith("sum") for o in obs)


def test_shape_dtype_and_layer_index_are_recovered_from_a_line():
    obs = {o.name: o for o in parse_eval_callback(_dump(n_layers=1))}
    topk = obs["ffn_moe_topk-0"]
    assert topk.ne == (8, 5, 1, 1)
    assert topk.dtype == "i32"
    assert topk.op == "view"
    assert topk.il == 0
    assert topk.base == "ffn_moe_topk"


def test_a_graph_level_node_has_no_layer_index():
    obs = {o.name: o for o in parse_eval_callback(_dump(n_layers=1))}
    assert obs["inp_embd"].il is None
    assert obs["inp_embd"].base == "inp_embd"


def test_shape_fragments_in_the_source_list_are_not_mistaken_for_source_names():
    """`src0{2048, 5, 1, 1}, }` splits on ',' into four pieces; three are shape, not sources."""
    obs = parse_eval_callback(
        _line("ffn_moe_probs-0", "f32", "soft_max", (64, 5, 1, 1),
              src0="ffn_moe_logits-0", src0_ne=(64, 5, 1, 1))
    )
    assert obs[0].src_names == ("ffn_moe_logits-0",)


def test_an_empty_or_unrelated_log_yields_no_observations():
    assert parse_eval_callback("") == []
    assert parse_eval_callback("llama_model_loader: - kv 0: general.architecture str = olmoe") == []


def test_inventory_counts_occurrences_rather_than_collapsing_them():
    inv = inventory_by_layer(parse_eval_callback(_dump(n_layers=2, duplicate="ffn_moe_topk")))
    assert inv[0].counts["ffn_moe_topk"] == 2
    assert inv[1].counts["ffn_moe_topk"] == 2


# --- the gate --------------------------------------------------------------------------------


def test_a_healthy_graph_confirms_and_resolves_all_three_names():
    r = confirm_nodes(CFG, parse_eval_callback(_dump()))
    assert r.confirmed, r.errors
    assert r.node_topk == "ffn_moe_topk-%d"
    assert r.node_logits == "ffn_moe_probs-%d"
    assert r.node_router_input == "ffn_norm-%d"
    assert r.selection_node == "ffn_moe_probs"
    assert r.moe_layers == [0, 1, 2, 3]


@pytest.mark.parametrize("dup", ["ffn_moe_topk", "ffn_moe_probs"])
def test_a_duplicated_node_fails_the_gate(dup):
    """The failure this module exists for: two cb() hits per layer, right-sized wrong-ordered rows."""
    r = confirm_nodes(CFG, parse_eval_callback(_dump(duplicate=dup)))
    assert not r.confirmed
    assert any("expected exactly 1" in e for e in r.errors)
    assert any(dup in e for e in r.errors)


@pytest.mark.parametrize("missing", ["ffn_moe_topk", "ffn_moe_probs"])
def test_a_missing_node_fails_the_gate(missing):
    r = confirm_nodes(CFG, parse_eval_callback(_dump(omit=missing)))
    assert not r.confirmed
    assert any("no node named" in e for e in r.errors)


def test_a_graph_with_fewer_layers_than_the_config_claims_fails():
    """A config saying 4 MoE layers against a 3-layer graph must not silently trace 3."""
    r = confirm_nodes(CFG, parse_eval_callback(_dump(n_layers=3)))
    assert not r.confirmed
    assert any("layer 3" in e for e in r.errors)


def test_a_node_later_in_the_selection_chain_than_predicted_fails_the_gate():
    """If `ffn_moe_probs_biased` exists, top-k consumed it -- the predicted node's margins are not
    the margins that decided the routing, and nothing about the file sizes would reveal that."""
    dump = _dump(extra_chain="ffn_moe_probs_biased")
    r = confirm_nodes(CFG, parse_eval_callback(dump))
    assert not r.confirmed
    assert any("LATER in the selection chain" in e for e in r.errors)


def test_a_correctly_predicted_biased_node_is_selected_rather_than_flagged():
    """Same graph, but a config that declares the expert-selection bias: now it is expected."""
    cfg = dict(CFG, has_expert_selection_bias=True)
    r = confirm_nodes(cfg, parse_eval_callback(_dump(extra_chain="ffn_moe_probs_biased")))
    assert r.confirmed, r.errors
    assert r.selection_node == "ffn_moe_probs_biased"
    assert r.node_logits == "ffn_moe_probs_biased-%d"


# --- shape and dtype checks (a right name on the wrong tensor) --------------------------------


def test_a_topk_node_whose_leading_dim_is_not_top_k_fails():
    """The name-only check's blind spot: ne[0]=64 means this is the argsort, not the top-k view."""
    r = confirm_nodes(CFG, parse_eval_callback(_dump(top_k=64)))
    assert not r.confirmed
    assert any("expected top_k=8" in e for e in r.errors)


def test_a_logits_node_whose_leading_dim_is_not_n_experts_fails():
    r = confirm_nodes(CFG, parse_eval_callback(_dump(n_experts=32)))
    assert not r.confirmed
    assert any("expected n_experts=64" in e for e in r.errors)


def test_a_router_input_whose_leading_dim_is_not_hidden_dim_fails():
    r = confirm_nodes(CFG, parse_eval_callback(_dump(hidden=1024)))
    assert not r.confirmed
    assert any("expected hidden_dim=2048" in e for e in r.errors)


def test_a_float_topk_node_fails_because_expert_indices_are_i32():
    r = confirm_nodes(CFG, parse_eval_callback(_dump(topk_dtype="f32")))
    assert not r.confirmed
    assert any("expected i32" in e for e in r.errors)


# --- router input resolution -------------------------------------------------------------------


def test_the_router_input_is_resolved_from_the_logits_node_sources_not_from_its_name():
    """An architecture that names its pre-router norm something nobody has seen still resolves,
    because `ffn_moe_logits` is `mul_mat(ffn_gate_inp, cur)` and names `cur` outright."""
    r = confirm_nodes(CFG, parse_eval_callback(_dump(router_input="some_unknown_norm")))
    assert r.confirmed, r.errors
    assert r.node_router_input == "some_unknown_norm-%d"
    assert any("structurally" in n for n in r.notes)


def test_the_router_weight_is_never_mistaken_for_the_router_input():
    """Both operands of the mul_mat are sources; picking src[0] blindly would name the weight."""
    r = confirm_nodes(CFG, parse_eval_callback(_dump()))
    assert r.confirmed, r.errors
    assert "ffn_gate_inp" not in r.node_router_input
    assert r.node_router_input == "ffn_norm-%d"


def test_a_router_input_that_differs_between_layers_fails_rather_than_picking_one():
    """The probes would be conditioned on a mixture, and the resulting underperformance is
    indistinguishable from the study's actual finding about predictability."""
    r = confirm_nodes(
        CFG,
        parse_eval_callback(_dump(router_input_of_layer=lambda il: "ffn_norm" if il < 2 else "ffn_inp")),
    )
    assert not r.confirmed
    assert any("not the same node in every MoE layer" in e for e in r.errors)


def test_without_source_information_the_scan_falls_back_to_a_name_and_says_so():
    r = confirm_nodes(CFG, parse_eval_callback(_dump(logits_srcs=False)))
    assert r.confirmed, r.errors
    assert r.node_router_input == "ffn_norm-%d"
    assert any("name fallback" in n for n in r.notes)


def test_the_name_fallback_never_settles_for_attn_norm():
    """`attn_norm` exists in every layer of every transformer, so a fallback list containing it
    could never fail -- and a guess that cannot fail is not a check."""
    r = confirm_nodes(
        CFG, parse_eval_callback(_dump(router_input="some_unknown_norm", logits_srcs=False))
    )
    assert not r.confirmed
    assert any("could not be resolved" in e for e in r.errors)
    assert "attn_norm" not in r.node_router_input


def test_an_explicitly_forced_router_input_is_used_and_still_checked():
    dump = _dump(router_input="ffn_inp")
    r = confirm_nodes(CFG, parse_eval_callback(dump), router_input_base="ffn_inp")
    assert r.confirmed, r.errors
    bad = confirm_nodes(CFG, parse_eval_callback(dump), router_input_base="ffn_norm")
    assert not bad.confirmed


# --- config-level refusals ----------------------------------------------------------------------


def test_an_interleaved_moe_layer_map_is_refused_before_any_scanning():
    """`moe_layer_freq != 1` makes offset+range the wrong map; nodespec refuses and so must this."""
    with pytest.raises(NodeSpecError, match="moe_layer_freq"):
        confirm_nodes(dict(CFG, moe_layer_freq=2), parse_eval_callback(_dump()))


def test_an_architecture_that_routes_outside_the_chain_is_refused():
    with pytest.raises(NodeSpecError, match="llama4"):
        confirm_nodes(dict(CFG, arch="llama4"), parse_eval_callback(_dump()))


def test_a_nonzero_layer_offset_shifts_which_il_values_are_checked():
    """DeepSeek-V2-Lite's layer 0 is dense: trace layer 0 is il=1."""
    cfg = dict(CFG, moe_layer_offset=1, n_moe_layers=3)
    r = confirm_nodes(cfg, parse_eval_callback(_dump(n_layers=4)))
    assert r.confirmed, r.errors
    assert r.moe_layers == [1, 2, 3]


def test_an_empty_log_is_an_error_and_never_a_confirmation():
    r = confirm_nodes(CFG, [])
    assert not r.confirmed
    assert any("no eval-callback node lines" in e for e in r.errors)


# --- CLI ------------------------------------------------------------------------------------


def _models_yaml(tmp_path, **overrides):
    import yaml
    cfg = {k: v for k, v in CFG.items() if k != "key"}
    cfg.update(overrides)
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"models": {"olmoe-test": cfg}}), encoding="utf-8")
    return p


def test_the_cli_parses_a_saved_log_without_loading_anything(tmp_path, capsys):
    log = tmp_path / "dump.txt"
    log.write_text(_dump(), encoding="utf-8")
    rc = main(["olmoe-test", "--models", str(_models_yaml(tmp_path)), "--from-log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CONFIRMED" in out
    assert "logit_tensor_used: ffn_moe_probs" in out
    assert "ffn_moe_topk-%d" in out


def test_the_cli_exits_nonzero_on_a_failed_gate(tmp_path, capsys):
    log = tmp_path / "dump.txt"
    log.write_text(_dump(duplicate="ffn_moe_topk"), encoding="utf-8")
    rc = main(["olmoe-test", "--models", str(_models_yaml(tmp_path)), "--from-log", str(log)])
    assert rc == 1
    assert "HALT GATE" in capsys.readouterr().err


def test_the_cli_rejects_an_unknown_model_key(tmp_path, capsys):
    log = tmp_path / "dump.txt"
    log.write_text(_dump(), encoding="utf-8")
    rc = main(["nope", "--models", str(_models_yaml(tmp_path)), "--from-log", str(log)])
    assert rc == 1
    assert "not in" in capsys.readouterr().err


def test_run_and_from_log_are_mutually_exclusive_so_a_model_load_is_never_implicit(tmp_path):
    with pytest.raises(SystemExit):
        main(["olmoe-test", "--from-log", "x", "--run"])
    with pytest.raises(SystemExit):
        main(["olmoe-test"])  # neither: must be explicit


def test_run_requires_an_explicit_gguf_path(tmp_path):
    with pytest.raises(SystemExit):
        main(["olmoe-test", "--models", str(_models_yaml(tmp_path)), "--run"])
