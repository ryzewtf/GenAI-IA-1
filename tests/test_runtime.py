"""Runtime tests — config hashing, shard ledger, session budget, scratch preflight.

These cover the machinery plan S.3 calls the resumable runner contract. The theme throughout:
a run that dies must leave state that is either complete or absent, never partial.
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.runtime.config import (
    ConfigError,
    IncompatibleShardError,
    RunConfig,
    assert_shards_compatible,
    canonical_hash,
)
from src.runtime.preflight import (
    PreflightError,
    check_upload_credentials,
    from_config as preflight_from_config,
    run_preflight,
)
from src.runtime.session import SessionBudget
from src.runtime.state import ShardRecord, ShardState, StateError

REPO_RUN_CONFIG = "configs/run.yaml"


# -- config hashing --------------------------------------------------------------------------


def test_repo_run_config_loads():
    config = RunConfig.load(REPO_RUN_CONFIG)
    assert config.inference["ctx_size"] == 2048, "invariant I4: -c 2048 everywhere"
    assert config.inference["split_mode"] == "layer"
    assert config.analysis["clamp_mi_at_zero"] is False, "invariant I8"
    assert config.build["ggml_native"] is False, "T0.2: GGML_NATIVE=OFF or SIGILL"
    assert len(config.sha256) == 64


def test_hash_is_stable_across_key_order():
    a = canonical_hash({"x": 1, "y": {"b": 2, "a": 3}})
    b = canonical_hash({"y": {"a": 3, "b": 2}, "x": 1})
    assert a == b


def test_hash_changes_when_a_pinned_flag_changes(tmp_path):
    base = yaml.safe_load(open(REPO_RUN_CONFIG, encoding="utf-8"))

    first = tmp_path / "a.yaml"
    first.write_text(yaml.safe_dump(base), encoding="utf-8")

    bumped = json.loads(json.dumps(base))
    bumped["hashed"]["inference"]["ubatch_size"] = 256
    second = tmp_path / "b.yaml"
    second.write_text(yaml.safe_dump(bumped), encoding="utf-8")

    assert RunConfig.load(first).sha256 != RunConfig.load(second).sha256


def test_hash_ignores_operational_settings(tmp_path):
    """Bumping an upload retry must not invalidate every shard already collected."""
    base = yaml.safe_load(open(REPO_RUN_CONFIG, encoding="utf-8"))

    first = tmp_path / "a.yaml"
    first.write_text(yaml.safe_dump(base), encoding="utf-8")

    tweaked = json.loads(json.dumps(base))
    tweaked["unhashed"]["preflight"]["min_write_mbps"] = 999
    second = tmp_path / "b.yaml"
    second.write_text(yaml.safe_dump(tweaked), encoding="utf-8")

    assert RunConfig.load(first).sha256 == RunConfig.load(second).sha256


def test_platform_is_inside_the_hash(tmp_path):
    """Invariant I3 — sm_75 and sm_120 traces are different experiments."""
    base = yaml.safe_load(open(REPO_RUN_CONFIG, encoding="utf-8"))

    first = tmp_path / "t4.yaml"
    first.write_text(yaml.safe_dump(base), encoding="utf-8")

    local = json.loads(json.dumps(base))
    local["hashed"]["platform"]["gpu_arch"] = 120
    second = tmp_path / "blackwell.yaml"
    second.write_text(yaml.safe_dump(local), encoding="utf-8")

    assert RunConfig.load(first).sha256 != RunConfig.load(second).sha256


def test_collection_readiness_catches_the_expensive_mistakes(tmp_path):
    base = yaml.safe_load(open(REPO_RUN_CONFIG, encoding="utf-8"))

    # An unpinned build must be refused. Constructed rather than taken from the shipped config:
    # run.yaml now carries a real commit, and a test that only passes while a field happens to be
    # null stops testing anything the moment the field is filled in.
    unpinned = json.loads(json.dumps(base))
    unpinned["hashed"]["build"]["llama_cpp_commit"] = None
    path = tmp_path / "unpinned.yaml"
    path.write_text(yaml.safe_dump(unpinned), encoding="utf-8")
    with pytest.raises(ConfigError, match="llama_cpp_commit"):
        RunConfig.load(path).assert_collection_ready()

    # The shipped config must be collection-ready as-is: pinning the commit was the last
    # outstanding field, so a regression here means someone nulled a hashed knob.
    RunConfig.load(REPO_RUN_CONFIG).assert_collection_ready()

    ready = json.loads(json.dumps(base))
    ready["hashed"]["build"]["llama_cpp_commit"] = "deadbeef"
    good = tmp_path / "ready.yaml"
    good.write_text(yaml.safe_dump(ready), encoding="utf-8")
    RunConfig.load(good).assert_collection_ready()  # must not raise

    for mutate, pattern in [
        (lambda c: c["hashed"]["inference"].__setitem__("tensor_split", None), "tensor_split"),
        (lambda c: c["hashed"]["inference"].__setitem__("flash_attn", None), "flash_attn"),
        (lambda c: c["hashed"]["analysis"].__setitem__("clamp_mi_at_zero", True), "clamp"),
        (lambda c: c["hashed"]["capture"].__setitem__("streams", ["logits"]), "topk"),
        (lambda c: c["hashed"]["build"].__setitem__("ggml_native", True), "ggml_native"),
        (
            lambda c: c["hashed"]["capture"].__setitem__("clear_kv_between_docs", False),
            "clear_kv_between_docs",
        ),
    ]:
        broken = json.loads(json.dumps(ready))
        mutate(broken)
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        with pytest.raises(ConfigError, match=pattern):
            RunConfig.load(path).assert_collection_ready()


def test_assert_shards_compatible_reports_every_conflict():
    a = {"shard_id": 0, "run_config_sha256": "x", "quant": "Q4_K_M"}
    b = {"shard_id": 1, "run_config_sha256": "y", "quant": "Q8_0"}
    with pytest.raises(IncompatibleShardError) as excinfo:
        assert_shards_compatible([a, b], ("run_config_sha256", "quant"))
    message = str(excinfo.value)
    assert "run_config_sha256" in message and "quant" in message


# -- shard ledger -----------------------------------------------------------------------------


def _record(shard_id: int = 0, **kwargs) -> ShardRecord:
    defaults = dict(
        shard_id=shard_id,
        n_tokens=1000,
        n_captured=50,
        file_sha256={"topk.bin": "a" * 64},
        doc_range=(0, 10),
        upload_verified=True,
    )
    defaults.update(kwargs)
    return ShardRecord(**defaults)


def test_shard_is_not_complete_until_the_upload_is_verified(tmp_path):
    state = ShardState.load_or_create(tmp_path / "state.json", "m", "c", "h")
    with pytest.raises(StateError, match="upload round-trip"):
        state.mark_complete(_record(upload_verified=False))
    assert state.completed_ids() == set()


def test_shard_requires_checksums(tmp_path):
    state = ShardState.load_or_create(tmp_path / "state.json", "m", "c", "h")
    with pytest.raises(StateError, match="checksums"):
        state.mark_complete(_record(file_sha256={}))


def test_ledger_resumes_and_reports_pending(tmp_path):
    path = tmp_path / "state.json"
    state = ShardState.load_or_create(path, "m", "c", "h")
    state.mark_complete(_record(0))
    state.mark_complete(_record(2))

    resumed = ShardState.load_or_create(path, "m", "c", "h")
    assert resumed.completed_ids() == {0, 2}
    assert resumed.pending(range(5)) == [1, 3, 4]
    assert resumed.n_tokens == 2000


def test_ledger_refuses_to_resume_under_a_different_config(tmp_path):
    path = tmp_path / "state.json"
    ShardState.load_or_create(path, "m", "c", "hash-one").mark_complete(_record(0))
    with pytest.raises(StateError, match="cannot be merged"):
        ShardState.load_or_create(path, "m", "c", "hash-two")


def test_recollecting_a_shard_with_different_bytes_is_an_error(tmp_path):
    """If a re-run is not bit-identical, an unpinned variable is in play (plan T3.6)."""
    state = ShardState.load_or_create(tmp_path / "state.json", "m", "c", "h")
    state.mark_complete(_record(0))
    with pytest.raises(StateError, match="different checksums"):
        state.mark_complete(_record(0, file_sha256={"topk.bin": "b" * 64}))


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "state.json"
    state = ShardState.load_or_create(path, "m", "c", "h")
    state.mark_complete(_record(0))
    assert json.loads(path.read_text())["n_shards_complete"] == 1
    assert list(tmp_path.glob("*.tmp")) == []


# -- session budget ---------------------------------------------------------------------------


def test_budget_stops_inside_the_reserve_window():
    budget = SessionBudget(wall_limit_s=100, reserve_s=30)
    assert budget.usable_s == 70
    assert not budget.should_stop()
    budget._start -= 75  # simulate 75 s elapsed
    assert budget.should_stop()
    assert budget.remaining() == pytest.approx(25, abs=1)


def test_budget_rejects_a_reserve_that_leaves_no_working_time():
    with pytest.raises(ValueError, match="reserve_s"):
        SessionBudget(wall_limit_s=100, reserve_s=100)


def test_budget_reads_the_unhashed_session_block():
    budget = SessionBudget.from_config(RunConfig.load(REPO_RUN_CONFIG))
    assert budget.wall_limit_s == 43200
    assert budget.reserve_s == 1800


# -- preflight --------------------------------------------------------------------------------


def test_preflight_measures_and_passes_on_a_small_probe(tmp_path):
    result = run_preflight(tmp_path, probe_bytes=4 * 1024 * 1024, min_write_mbps=0.01)
    assert result.passed
    assert result.write_mbps > 0
    assert list(tmp_path.glob("preflight_*")) == [], "probe file must be deleted"


def test_preflight_fails_loudly_when_throughput_is_below_the_floor(tmp_path):
    with pytest.raises(PreflightError, match="below"):
        run_preflight(tmp_path, probe_bytes=1024 * 1024, min_write_mbps=1e9)


def test_preflight_writes_its_report(tmp_path):
    report = tmp_path / "out" / "preflight.json"
    run_preflight(
        tmp_path, probe_bytes=1024 * 1024, min_write_mbps=0.01, report_path=report
    )
    assert json.loads(report.read_text())["passed"] is True


def test_preflight_rejects_a_mount_too_small_for_the_collection(tmp_path):
    """T0.1 regression. The throughput probe is 2 GB and passes just as happily on Kaggle's
    19.5 GiB overlay as on the 1 TB scratch disk, so it cannot tell them apart. Only a
    free-space floor can, and it has to be checked before the probe runs."""
    with pytest.raises(PreflightError, match="below the .* GB floor"):
        run_preflight(
            tmp_path, probe_bytes=1024 * 1024, min_write_mbps=0.01, min_free_gb=1e9
        )


def test_preflight_says_when_it_had_to_create_the_scratch_directory(tmp_path):
    """The exact T0.1 failure: `unhashed.paths.scratch` named /kaggle/temp, which does not
    exist on the current image. mkdir() would have created it on the wrong filesystem and every
    check downstream would have passed. Creating the directory is a diagnosis, not a detail."""
    missing = tmp_path / "not" / "there"
    result = run_preflight(
        missing, probe_bytes=1024 * 1024, min_write_mbps=0.01, min_free_gb=1e9,
        raise_on_fail=False,
    )
    assert result.created_scratch is True
    assert "did not exist" in result.detail
    assert "paths.scratch" in result.detail

    again = run_preflight(missing, probe_bytes=1024 * 1024, min_write_mbps=0.01)
    assert again.created_scratch is False, "a directory that already existed is not a red flag"


def test_the_repo_scratch_path_is_not_the_mount_t0_1_found_missing():
    scratch = RunConfig.load(REPO_RUN_CONFIG).unhashed["paths"]["scratch"]
    assert scratch != "/kaggle/temp", (
        "T0.1 (2026-08-18) found /kaggle/temp absent from the Kaggle image; it is not a "
        "usable scratch mount"
    )


def test_hf_upload_backend_requires_a_token_before_collection_not_after(monkeypatch):
    """T0.1 reported both token variables unset on a fresh session while upload.backend is 'hf'.
    Kaggle Secrets are opt-in, so unset is the DEFAULT state. Discovering it in upload.py means
    discovering it after the traces exist, on a scratch disk that dies with the session."""
    config = RunConfig.load(REPO_RUN_CONFIG)
    assert config.unhashed["upload"]["backend"] == "hf"

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(PreflightError, match="HF_TOKEN"):
        check_upload_credentials(config)
    with pytest.raises(PreflightError, match="HF_TOKEN"):
        preflight_from_config(config)

    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    check_upload_credentials(config)  # must not raise


def test_credential_gate_stays_out_of_the_way_of_other_backends(monkeypatch, tmp_path):
    raw = yaml.safe_load(open(REPO_RUN_CONFIG, encoding="utf-8"))
    raw["unhashed"]["upload"]["backend"] = "local"
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    check_upload_credentials(RunConfig.load(path))  # must not raise
