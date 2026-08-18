"""Upload round-trip tests — plan S.3 step d, T5.3, T3.6.

The theme, as in ``test_runtime.py``: a run that dies must leave state that is either complete or
absent, never partial. Here the specific hazard is an upload that *reports success* while storing
truncated or corrupt bytes — the plan's risk table rates it medium-likelihood, high-impact and
silent. The corruption tests below are the reason this file exists; everything else is scaffolding
around them.

Everything runs offline through :class:`LocalDirBackend`, which is a shipped backend rather than a
mock, so these tests exercise the same round-trip code path a Kaggle session does.
``huggingface_hub`` is not installed in this venv and nothing here imports it.
"""

from __future__ import annotations

import builtins
import hashlib

import pytest

from src.runtime.state import ShardState, StateError
from src.runtime.upload import (
    HF_TOKEN_ENV_VARS,
    HFBackend,
    LocalDirBackend,
    UploadError,
    load_remote_manifest,
    sha256_file,
    upload_shard,
    verify_remote_shard,
)
from src.traces.format import MANIFEST_NAME, STREAM_FILES, TraceSpec
from src.traces.synth import make_synthetic_trace

REMOTE_PREFIX = "traces/synth-moe/synth-v1/shard_00000"
SPEC = TraceSpec(n_moe_layers=2, n_experts=8, top_k=2, hidden_dim=4)


# -- fixtures ---------------------------------------------------------------------------------


def _make_shard(tmp_path, *, seed: int = 0, name: str = "local"):
    """One small real shard directory (a few KB) written by the synth generator."""
    trace = make_synthetic_trace(
        tmp_path / name,
        spec=SPEC,
        shard_sizes=(20,),
        seed=seed,
    )
    return trace.root / trace.model / trace.corpus / "shard_00000"


@pytest.fixture
def shard(tmp_path):
    return _make_shard(tmp_path)


@pytest.fixture
def backend(tmp_path):
    return LocalDirBackend(tmp_path / "remote")


def _ledger(tmp_path):
    return ShardState.load_or_create(
        tmp_path / "state.json", "synth-moe", "synth-v1", "0" * 64
    )


class TruncatingBackend(LocalDirBackend):
    """Stores every byte but the last of one named file. Models a dropped final chunk."""

    def __init__(self, root, victim: str):
        super().__init__(root)
        self.victim = victim

    def upload_file(self, local_path, remote_path):
        super().upload_file(local_path, remote_path)
        if remote_path.endswith(self.victim):
            target = self._resolve(remote_path)
            target.write_bytes(target.read_bytes()[:-1])


class BitFlipBackend(LocalDirBackend):
    """Stores the right *number* of bytes with one byte wrong — size checks cannot see this."""

    def __init__(self, root, victim: str):
        super().__init__(root)
        self.victim = victim

    def upload_file(self, local_path, remote_path):
        super().upload_file(local_path, remote_path)
        if remote_path.endswith(self.victim):
            target = self._resolve(remote_path)
            blob = bytearray(target.read_bytes())
            blob[0] ^= 0x01
            target.write_bytes(bytes(blob))


class FlakyBackend(LocalDirBackend):
    """Raises on the Nth upload, leaving the earlier files present — a killed session."""

    def __init__(self, root, fail_after: int):
        super().__init__(root)
        self.fail_after = fail_after
        self.calls = 0

    def upload_file(self, local_path, remote_path):
        self.calls += 1
        if self.calls > self.fail_after:
            raise UploadError("simulated session kill mid-upload")
        super().upload_file(local_path, remote_path)


# -- sha256_file ------------------------------------------------------------------------------


def test_sha256_file_matches_whole_file_hash_across_chunks(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(range(256)) * 1000)  # 256 kB, several chunks at chunk_bytes=1024
    assert path.stat().st_size > 1024

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path, chunk_bytes=1024) == expected
    assert sha256_file(path) == expected


def test_sha256_file_notices_a_single_flipped_byte(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00" * 4096)
    before = sha256_file(path, chunk_bytes=64)
    path.write_bytes(b"\x01" + b"\x00" * 4095)
    assert sha256_file(path, chunk_bytes=64) != before


# -- happy path -------------------------------------------------------------------------------


def test_clean_upload_verifies_and_reports_per_file_hashes(shard, backend):
    result = upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX)

    assert result.verified is True
    assert result.shard_id == 0
    assert result.bytes_uploaded == sum(m["size"] for m in result.files.values())
    assert result.elapsed_s >= 0.0

    for name in list(STREAM_FILES.values()) + [MANIFEST_NAME]:
        assert name in result.files, name
        local = shard / name
        assert result.files[name]["size"] == local.stat().st_size
        assert result.files[name]["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()

    # manifest.json is deliberately absent from the ledger checksum map: it carries
    # collected_utc, so a bit-identical recollection would otherwise look like a T3.6 conflict.
    assert MANIFEST_NAME not in result.file_sha256
    assert set(result.file_sha256) == set(STREAM_FILES.values())

    payload = result.to_json()
    assert payload["verified"] is True
    assert payload["remote_prefix"] == REMOTE_PREFIX


def test_upload_marks_the_shard_complete_in_the_ledger(tmp_path, shard, backend):
    state = _ledger(tmp_path)
    result = upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX, state=state)

    assert state.completed_ids() == {0}
    record = state.record(0)
    assert record.upload_verified is True
    assert record.file_sha256 == result.file_sha256
    assert record.n_tokens == 20
    assert state.path.exists()


def test_remote_manifest_is_readable_after_upload(shard, backend):
    upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX)
    manifest = load_remote_manifest(backend, remote_prefix=REMOTE_PREFIX)
    assert manifest is not None and manifest["shard_id"] == 0
    assert load_remote_manifest(backend, remote_prefix="traces/nope") is None


def test_verify_remote_shard_reverifies_without_reuploading(shard, backend):
    upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX)
    report = verify_remote_shard(shard, backend, remote_prefix=REMOTE_PREFIX)
    assert report["verified"] is True
    assert report["missing"] == [] and report["failures"] == []


def test_verify_remote_shard_reports_a_never_uploaded_shard(shard, backend):
    report = verify_remote_shard(shard, backend, remote_prefix=REMOTE_PREFIX)
    assert report["verified"] is False
    assert set(report["missing"]) == set(list(STREAM_FILES.values()) + [MANIFEST_NAME])


# -- THE point of the task: corruption is caught ----------------------------------------------


@pytest.mark.parametrize("victim", ["topk.bin", "logits.bin", "hidden.bin", "tokens.bin"])
def test_truncated_upload_is_caught_and_names_the_file(tmp_path, shard, victim):
    """A "successful" upload that stored one byte short must fail HERE, not in Phase 9."""
    bad = TruncatingBackend(tmp_path / "remote", victim)

    with pytest.raises(UploadError) as excinfo:
        upload_shard(shard, bad, remote_prefix=REMOTE_PREFIX)

    message = str(excinfo.value)
    assert victim in message
    assert "-1" in message, f"the diagnostic should name the byte delta: {message}"


def test_corrupt_upload_with_matching_size_is_caught_by_the_hash(tmp_path, shard):
    bad = BitFlipBackend(tmp_path / "remote", "topk.bin")

    with pytest.raises(UploadError) as excinfo:
        upload_shard(shard, bad, remote_prefix=REMOTE_PREFIX)

    message = str(excinfo.value)
    assert "topk.bin" in message
    local_hash = hashlib.sha256((shard / "topk.bin").read_bytes()).hexdigest()
    assert local_hash in message, "both hashes must be in the diagnostic"
    assert message.count("sha256") >= 2


def test_failed_verification_leaves_the_ledger_untouched(tmp_path, shard):
    state = _ledger(tmp_path)
    bad = TruncatingBackend(tmp_path / "remote", "logits.bin")

    with pytest.raises(UploadError):
        upload_shard(shard, bad, remote_prefix=REMOTE_PREFIX, state=state)

    assert state.completed_ids() == set()
    assert not state.path.exists(), "a failed upload must not even write a ledger file"

    reloaded = ShardState.load_or_create(state.path, "synth-moe", "synth-v1", "0" * 64)
    assert reloaded.pending([0]) == [0], "the shard must be recollected next session"


def test_local_files_survive_a_failed_verification(tmp_path, shard):
    bad = TruncatingBackend(tmp_path / "remote", "hidden.bin")
    before = {name: (shard / name).read_bytes() for name in STREAM_FILES.values()}

    with pytest.raises(UploadError):
        upload_shard(
            shard, bad, remote_prefix=REMOTE_PREFIX, delete_local_on_success=True
        )

    for name, blob in before.items():
        assert (shard / name).read_bytes() == blob, f"{name} was deleted on a failed upload"
    assert (shard / MANIFEST_NAME).exists()


def test_local_files_are_deleted_only_on_success(tmp_path, shard, backend):
    state = _ledger(tmp_path)
    result = upload_shard(
        shard,
        backend,
        remote_prefix=REMOTE_PREFIX,
        delete_local_on_success=True,
        state=state,
    )

    assert result.verified is True
    for name in list(STREAM_FILES.values()) + [MANIFEST_NAME]:
        assert not (shard / name).exists(), f"{name} should have been deleted (I9, T4.4)"
    # The remote copy is the surviving one, and it is intact.
    assert backend.exists(f"{REMOTE_PREFIX}/topk.bin")
    assert state.record(0).upload_verified is True


def test_delete_without_verify_is_refused(shard, backend):
    with pytest.raises(UploadError, match="requires verify=True"):
        upload_shard(
            shard,
            backend,
            remote_prefix=REMOTE_PREFIX,
            verify=False,
            delete_local_on_success=True,
        )
    assert (shard / "topk.bin").exists()


def test_unverified_upload_cannot_be_recorded(tmp_path, shard, backend):
    state = _ledger(tmp_path)
    with pytest.raises(UploadError, match="unverified"):
        upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX, verify=False, state=state)
    assert state.completed_ids() == set()


def test_locally_truncated_shard_is_refused_before_any_upload(shard, backend):
    """A short file on disk is a capture bug, not a transfer bug; fail before spending bandwidth."""
    victim = shard / "logits.bin"
    victim.write_bytes(victim.read_bytes()[:-4])

    with pytest.raises(UploadError, match="manifest arithmetic"):
        upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX)

    assert backend.list_files(REMOTE_PREFIX) == []


# -- resumability and T3.6 -------------------------------------------------------------------


def test_reuploading_identical_content_is_idempotent(tmp_path, shard, backend):
    state = _ledger(tmp_path)
    first = upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX, state=state)
    second = upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX, state=state)

    assert second.verified is True
    assert second.file_sha256 == first.file_sha256
    assert state.completed_ids() == {0}
    assert sorted(backend.list_files(REMOTE_PREFIX)) == sorted(
        f"{REMOTE_PREFIX}/{n}" for n in list(STREAM_FILES.values()) + [MANIFEST_NAME]
    )


def test_recollecting_a_shard_with_different_content_is_a_hard_error(tmp_path, backend):
    """Plan T3.6: a recollection that is not bit-identical means an unpinned variable."""
    state = _ledger(tmp_path)

    first = _make_shard(tmp_path, seed=0, name="run_a")
    upload_shard(first, backend, remote_prefix=REMOTE_PREFIX, state=state)

    different = _make_shard(tmp_path, seed=99, name="run_b")
    assert (different / "topk.bin").read_bytes() != (first / "topk.bin").read_bytes()

    with pytest.raises(StateError, match="different checksums"):
        upload_shard(different, backend, remote_prefix=REMOTE_PREFIX, state=state)

    # The ledger still holds the original evidence, unmodified.
    assert state.record(0).file_sha256 == {
        name: sha256_file(first / name) for name in STREAM_FILES.values()
    }


def test_interrupted_upload_leaves_a_clean_ledger_and_is_retryable(tmp_path, shard):
    state = _ledger(tmp_path)
    root = tmp_path / "remote"
    flaky = FlakyBackend(root, fail_after=2)

    with pytest.raises(UploadError, match="session kill"):
        upload_shard(shard, flaky, remote_prefix=REMOTE_PREFIX, state=state)

    assert state.completed_ids() == set()
    partial = LocalDirBackend(root).list_files(REMOTE_PREFIX)
    assert 0 < len(partial) < len(STREAM_FILES) + 1, "expected a genuinely partial remote copy"
    # Presence is not integrity: the partial remote must not read as verified.
    assert verify_remote_shard(shard, LocalDirBackend(root), remote_prefix=REMOTE_PREFIX)[
        "verified"
    ] is False

    # Retry over the same prefix: every file is re-sent, so the partial state heals.
    retried = upload_shard(
        shard, LocalDirBackend(root), remote_prefix=REMOTE_PREFIX, state=state
    )
    assert retried.verified is True
    assert state.completed_ids() == {0}


def test_a_shard_truncated_remotely_after_upload_fails_reverification(tmp_path, shard, backend):
    upload_shard(shard, backend, remote_prefix=REMOTE_PREFIX)

    remote_topk = tmp_path / "remote" / REMOTE_PREFIX / "topk.bin"
    remote_topk.write_bytes(remote_topk.read_bytes()[:-8])

    report = verify_remote_shard(shard, backend, remote_prefix=REMOTE_PREFIX)
    assert report["verified"] is False
    assert [f["file"] for f in report["failures"]] == ["topk.bin"]
    assert "topk.bin" in report["failures"][0]["reason"]


# -- backend hygiene --------------------------------------------------------------------------


def test_local_backend_rejects_a_traversing_remote_path(tmp_path, shard):
    backend = LocalDirBackend(tmp_path / "remote")
    with pytest.raises(UploadError, match="escapes"):
        upload_shard(shard, backend, remote_prefix="../../escape")


def test_hf_backend_raises_a_clear_error_when_the_library_is_absent(monkeypatch, tmp_path):
    """This venv has no huggingface_hub; the failure must be actionable, not an ImportError."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_do_not_leak")

    hf = HFBackend("someone/moe-traces")
    with pytest.raises(UploadError, match="huggingface_hub is not installed"):
        hf.upload_file(tmp_path / "nope.bin", "x/y.bin")


def test_hf_backend_raises_a_clear_error_when_no_token_is_present(monkeypatch):
    for name in HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(UploadError) as excinfo:
        HFBackend._token()
    message = str(excinfo.value)
    for name in HF_TOKEN_ENV_VARS:
        assert name in message, "the error must say which env var to set"


def test_no_token_value_ever_appears_in_an_exception_or_repr(monkeypatch, tmp_path):
    secret = "hf_zzz_super_secret_token_value"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", secret)

    hf = HFBackend("someone/moe-traces", repo_type="dataset")
    assert secret not in repr(hf)
    assert secret not in str(vars(hf))

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for call in (
        lambda: hf.upload_file(tmp_path / "a.bin", "x/y.bin"),
        lambda: hf.download_file("x/y.bin", tmp_path / "a.bin"),
        lambda: hf.exists("x/y.bin"),
        lambda: hf.list_files("x/"),
    ):
        with pytest.raises(UploadError) as excinfo:
            call()
        assert secret not in str(excinfo.value)
        assert secret not in repr(excinfo.value)
