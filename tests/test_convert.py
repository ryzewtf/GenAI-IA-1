"""T1.1 conversion recipe -- the decisions, not the subprocesses.

Everything here is reachable without a checkpoint, a network, or a build tree, which is the point:
the recipe is a claim about the whole panel and it should be checkable on a laptop. The parts that
genuinely need bytes (`convert_hf_to_gguf.py`, `llama-quantize`) are exercised on Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.kaggle_convert import (
    PANEL_ORDER,
    ConversionPlan,
    ConversionRecord,
    ConvertError,
    plan_conversion,
    resolve_model,
)
from src.capture.nodescan import NodeScanError, set_model_fields

MODELS_YAML = """\
schema_version: 1

defaults:
  quant: Q4_K_M

models:

  fresh:
    hf_repo: org/fresh
    # a comment that a PyYAML round trip would delete
    gguf: {repo: null, file: null, sha256: null, size_bytes: null}
    router_dtype: null

  recorded:
    hf_repo: org/recorded
    gguf: {repo: someone/else-GGUF,
           file: else.Q4_K_M.gguf,
           sha256: abc123,
           size_bytes: 42}
    router_dtype: F32
"""


@pytest.fixture()
def models_path(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    return path


# -- the recipe ------------------------------------------------------------------------------


def test_the_default_recipe_is_one_f16_intermediate_then_the_panel_quant():
    plan = plan_conversion("m", {"hf_repo": "org/m", "quant": "Q4_K_M"})
    assert (plan.outtype, plan.quantize_to) == ("f16", "Q4_K_M")
    assert plan.requantizes


def test_an_mxfp4_source_is_never_quantized():
    """Plan §0b names this by hand: 'GPT-OSS: native MXFP4, do not requantize'.

    The converter promotes the file type to MOSTLY_MXFP4_MOE off the source's own quant_method and
    repacks losslessly, so quantizing afterwards would be a lossy transform of an already-lossy
    format -- and Phase 8 exempts GPT-OSS from T8.4, so the error would be unmeasurable as well as
    unnecessary.
    """
    plan = plan_conversion("gpt-oss", {"hf_repo": "openai/gpt-oss-20b", "quant": "MXFP4"})
    assert plan.quantize_to is None
    assert not plan.requantizes
    assert "0b" in plan.reason


def test_mxfp4_is_recognised_case_insensitively():
    plan = plan_conversion("gpt-oss", {"hf_repo": "openai/gpt-oss-20b", "quant": "mxfp4"})
    assert plan.quantize_to is None


def test_the_outtype_is_carried_but_does_not_rescue_an_mxfp4_source():
    plan = plan_conversion("gpt-oss", {"hf_repo": "r", "quant": "MXFP4"}, outtype="bf16")
    assert plan.outtype == "bf16"
    assert plan.quantize_to is None


def test_a_model_with_no_hf_repo_is_an_error_not_a_guess():
    with pytest.raises(ConvertError, match="hf_repo"):
        plan_conversion("m", {"quant": "Q4_K_M"})


def test_every_panel_member_resolves_and_plans():
    """The recipe table must cover the real configs/models.yaml, not just a fixture."""
    for key in PANEL_ORDER:
        plan = plan_conversion(key, resolve_model(key))
        assert plan.hf_repo
        # Exactly one exception to the uniform recipe, and it is the documented one.
        assert (plan.quantize_to is None) == (key == "gpt-oss-20b")


def test_panel_order_is_the_whole_panel():
    from scripts.kaggle_convert import MODELS_CONFIG
    from src.runtime.setup_kaggle import _load_yaml

    assert set(PANEL_ORDER) == set(_load_yaml(MODELS_CONFIG)["models"])


# -- the record ------------------------------------------------------------------------------


def _record(tmp_path, **kw):
    plan = kw.pop("plan", ConversionPlan("m", "org/m", "f16", "Q4_K_M", "r"))
    return ConversionRecord(
        model_key=kw.pop("model_key", "m"),
        plan=plan,
        gguf_path=tmp_path / "m-Q4_K_M.gguf",
        size_bytes=123,
        sha256="d" * 64,
        llama_commit="7077abbe14c510cb829c93a1328c2815b5805ebd",
        **kw,
    )


def test_the_record_carries_the_converter_commit(tmp_path):
    """A shard manifest records only the GGUF's hash; this file is what maps it to a recipe."""
    d = _record(tmp_path).to_dict()
    assert d["llama_cpp_commit"].startswith("7077abbe")
    assert d["hf_repo"] == "org/m"
    assert d["sha256"] == "d" * 64


def test_the_yaml_block_declares_no_repo(tmp_path):
    block = _record(tmp_path).yaml_block()
    assert "repo: null" in block
    assert "file: m-Q4_K_M.gguf" in block
    assert "size_bytes: 123" in block


# -- writing it back -------------------------------------------------------------------------


def test_writing_a_gguf_block_preserves_comments(models_path):
    changes = set_model_fields(
        models_path, "fresh",
        {"gguf": "{repo: null, file: fresh-Q4_K_M.gguf, sha256: aa, size_bytes: 7}"},
    )
    assert changes
    text = models_path.read_text(encoding="utf-8")
    assert "a comment that a PyYAML round trip would delete" in text
    assert "file: fresh-Q4_K_M.gguf" in text


def test_an_all_null_flow_mapping_counts_as_unset(models_path):
    """`{repo: null, file: null, ...}` is what T1.1 has not run yet looks like, not a prior claim."""
    set_model_fields(models_path, "fresh", {"gguf": "{repo: null, file: x.gguf, sha256: b, size_bytes: 1}"})
    assert "file: x.gguf" in models_path.read_text(encoding="utf-8")


def test_overwriting_a_recorded_gguf_needs_force(models_path):
    """A different GGUF under the same key means every trace already collected is another
    experiment; that is a stop, not an overwrite."""
    with pytest.raises(NodeScanError, match="already records"):
        set_model_fields(models_path, "recorded", {"gguf": "{repo: null, file: mine.gguf}"})

    set_model_fields(models_path, "recorded", {"gguf": "{repo: null, file: mine.gguf}"}, force=True)
    text = models_path.read_text(encoding="utf-8")
    assert "file: mine.gguf" in text
    # The multi-line flow mapping it replaced must be gone whole -- an orphaned continuation line
    # leaves a file that parses fine right up until someone loads it.
    assert "else.Q4_K_M.gguf" not in text
    assert "size_bytes: 42" not in text
    import yaml
    assert yaml.safe_load(text)["models"]["recorded"]["gguf"]["file"] == "mine.gguf"


def test_a_rewrite_to_the_same_value_is_a_no_op(models_path):
    before = models_path.read_text(encoding="utf-8")
    assert set_model_fields(models_path, "recorded", {"router_dtype": "F32"}) == []
    assert models_path.read_text(encoding="utf-8") == before


# -- locating the converter ------------------------------------------------------------------


def _ctx(tmp_path, commit="7077abbe14c510cb829c93a1328c2815b5805ebd"):
    from src.runtime.setup_kaggle import SetupContext

    return SetupContext(
        scratch=tmp_path, dry_run=False, jobs=1, cuda_arch="75",
        llama_commit=commit, quant="Q4_K_M", models=(), hf_token_present=False,
    )


def test_the_session_checkout_is_preferred_over_the_vendor_dir(tmp_path):
    """On Kaggle it is the only one that exists: `.vendor/llama_cpp_pull` is a gitlink with no
    `.gitmodules`, so a fresh clone of this project leaves that directory EMPTY."""
    from scripts.kaggle_convert import find_llama_tree

    ctx = _ctx(tmp_path)
    ctx.llama_dir.mkdir(parents=True)
    (ctx.llama_dir / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    assert find_llama_tree(ctx) == ctx.llama_dir


def test_an_empty_checkout_is_not_mistaken_for_a_usable_one(tmp_path):
    from scripts.kaggle_convert import find_llama_tree

    ctx = _ctx(tmp_path)
    ctx.llama_dir.mkdir(parents=True)  # exists, but holds no converter
    with pytest.raises(ConvertError, match="gitlink"):
        find_llama_tree(ctx)


def test_a_checkout_at_the_wrong_commit_is_a_stop(tmp_path, monkeypatch):
    """A converter at another commit is another recipe -- the exact confound this script removes."""
    import scripts.kaggle_convert as kc

    ctx = _ctx(tmp_path)
    ctx.llama_dir.mkdir(parents=True)
    (ctx.llama_dir / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")

    class _Proc:
        stdout = "deadbeef" * 5 + "\n"

    monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(ConvertError, match="but configs/run.yaml pins"):
        kc.find_llama_tree(ctx)


def test_an_mxfp4_source_is_not_charged_for_an_f16_intermediate():
    """gpt-oss never grows one, and pretending it does refuses runs that would have fitted."""
    from scripts.kaggle_convert import estimate_peak_gb

    meta = {"approx_size_gb": 12.0}
    requant = estimate_peak_gb(meta, remote=False, requantizes=True)
    native = estimate_peak_gb(meta, remote=False, requantizes=False)
    assert native == 24.0  # source + repacked artifact, both at the native width
    assert requant > 3 * native  # an F16 intermediate of a 12 GiB Q4 file is ~44 GiB


def test_pruning_charges_the_largest_model_not_the_whole_panel(tmp_path, monkeypatch):
    import scripts.kaggle_convert as kc

    sizes = {"small": 4.0, "large": 18.0}
    monkeypatch.setattr(kc, "resolve_model", lambda k: {"approx_size_gb": sizes[k]})
    monkeypatch.setattr(kc, "plan_conversion", lambda k, m, **kw: ConversionPlan(
        model_key=k, hf_repo="x/y", outtype="f16", quantize_to="Q4_K_M", reason=""))

    keys = list(sizes)
    peaks = {k: kc.estimate_peak_gb({"approx_size_gb": v}, remote=False) for k, v in sizes.items()}

    monkeypatch.setattr(kc, "_free_gb", lambda p: peaks["large"] + 1)
    assert kc.check_space(keys, tmp_path, remote=False, prune=True) is None
    complaint = kc.check_space(keys, tmp_path, remote=False, prune=False)
    assert complaint is not None and "--prune" in complaint


# -- the arch pin applies to collection, not to conversion ------------------------------------


def _fake_smi(monkeypatch, compute_cap):
    """Make step_env see one GPU reporting `compute_cap`."""
    import src.runtime.setup_kaggle as sk

    class _Proc:
        stdout = f"NVIDIA Whatever, {compute_cap}, 16384 MiB\n"

    monkeypatch.setattr(sk.subprocess, "run", lambda *a, **k: _Proc())


def test_a_collecting_session_refuses_a_gpu_the_run_config_does_not_pin(tmp_path, monkeypatch):
    """I3: gpu_arch is inside run_config_sha256, so shards from another arch are another run."""
    from src.runtime.setup_kaggle import SetupError, step_env

    _fake_smi(monkeypatch, "12.0")
    with pytest.raises(SetupError, match="run_config_sha256"):
        step_env(_ctx(tmp_path))


def test_a_conversion_session_does_not_care_what_gpu_is_in_the_box(tmp_path, monkeypatch):
    """It compiles no kernels and writes no shards, so the local card is not a fact about the
    experiment -- and refusing here would only teach the operator to edit the pin."""
    import dataclasses

    from src.runtime.setup_kaggle import step_env

    _fake_smi(monkeypatch, "12.0")
    result = step_env(dataclasses.replace(_ctx(tmp_path), cuda=False))
    assert not result.failed
    assert "ignoring sm_75" in result.detail


# -- what environment produced the artifact ---------------------------------------------------


def test_the_record_carries_the_converter_environment():
    """gemma-4 cannot be converted under the transformers llama.cpp pins, so one checkpoint comes
    from a different library set. That has to be readable off the artifact, not off a stack trace."""
    plan = ConversionPlan(model_key="m", hf_repo="a/b", outtype="f16",
                          quantize_to="Q4_K_M", reason="r")
    record = ConversionRecord(
        model_key="m", plan=plan, gguf_path=Path("m-Q4_K_M.gguf"), size_bytes=1,
        sha256="ab" * 32, llama_commit="c" * 40,
        converter_env={"transformers": "5.15.1", "torch": "2.11.0+cpu"},
    )
    assert record.to_dict()["converter_env"]["transformers"] == "5.15.1"


def test_an_interpreter_that_cannot_report_versions_is_not_a_converter_env(tmp_path):
    from scripts.kaggle_convert import probe_converter_env

    with pytest.raises(ConvertError, match="working converter environment"):
        probe_converter_env(str(tmp_path / "no-such-python"))
