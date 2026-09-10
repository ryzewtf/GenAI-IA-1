"""T1.1 publication -- the bookkeeping, not the network.

Every check here is about the thing that makes an upload dangerous rather than slow: publishing a
file whose bytes no longer match the record that describes it, or forgetting which of eight
datasets already went up after the ninth failed.
"""

from __future__ import annotations

import json

import pytest

from scripts.kaggle_publish import (
    Artifact,
    PublishError,
    already_published,
    check_local,
    load_artifacts,
    register,
)

RECORD = {
    "file": "m-Q4_K_M.gguf",
    "sha256": "ab" * 32,
    "size_bytes": 10,
    "llama_cpp_commit": "c" * 40,
    "converter_env": {"transformers": "4.57.6"},
    "f16_sha256": None,
}


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A results/ and a gguf dir holding one 10-byte artifact called `m`."""
    import scripts.kaggle_publish as kp

    results = tmp_path / "results"
    results.mkdir()
    (results / "convert_m.json").write_text(json.dumps(RECORD), encoding="utf-8")
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "m-Q4_K_M.gguf").write_bytes(b"0123456789")
    monkeypatch.setattr(kp, "RESULTS", results)
    return results, gguf


def test_publication_is_driven_by_records_not_by_a_directory_listing(tmp_path, staged):
    _, gguf = staged
    (gguf / "stray-Q4_K_M.gguf").write_bytes(b"nobody asked for this")
    artifacts = load_artifacts(["m"], gguf)
    assert [a.path.name for a in artifacts] == ["m-Q4_K_M.gguf"]


def test_a_model_with_no_conversion_record_cannot_be_published(staged):
    _, gguf = staged
    with pytest.raises(PublishError, match="no conversion record"):
        load_artifacts(["never-converted"], gguf)


def test_the_kept_f16_gets_its_own_dataset(staged, monkeypatch):
    """T8.1 reads olmoe-0125 at two precisions, but only one of them is the model's artifact."""
    import scripts.kaggle_publish as kp

    _, gguf = staged
    (gguf / "m-F16.gguf").write_bytes(b"f16 bytes")
    monkeypatch.setattr(kp, "F16_EXTRAS", {"m": "m-F16.gguf"})

    artifacts = load_artifacts(["m"], gguf)
    assert [a.slug_suffix for a in artifacts] == ["m", "m-f16"]
    assert artifacts[1].is_f16 and artifacts[1].sha256 is None
    assert artifacts[1].handle("ryzewtf") == "ryzewtf/moe-panel/gguf/m-f16"


def test_a_truncated_file_is_not_published(staged):
    _, gguf = staged
    (gguf / "m-Q4_K_M.gguf").write_bytes(b"012")  # a conversion that died mid-write
    artifacts = load_artifacts(["m"], gguf)
    with pytest.raises(PublishError, match="models.yaml cannot describe"):
        check_local(artifacts[0], rehash=False)


def test_rehash_catches_a_file_that_changed_after_conversion(staged):
    _, gguf = staged
    (gguf / "m-Q4_K_M.gguf").write_bytes(b"9876543210")  # same length, different bytes
    artifacts = load_artifacts(["m"], gguf)
    with pytest.raises(PublishError, match="do not publish it"):
        check_local(artifacts[0], rehash=True)


def test_a_dry_run_does_not_read_thirteen_gigabytes_to_print_a_plan(staged, monkeypatch):
    import scripts.kaggle_publish as kp

    _, gguf = staged
    (gguf / "m-F16.gguf").write_bytes(b"f16 bytes")
    monkeypatch.setattr(kp, "F16_EXTRAS", {"m": "m-F16.gguf"})
    monkeypatch.setattr(kp, "_sha256", lambda p: pytest.fail("hashed during a dry run"))

    f16 = load_artifacts(["m"], gguf)[1]
    assert check_local(f16, rehash=False, dry_run=True) == ""


def test_a_rerun_after_one_failure_does_not_resend_the_seven_that_worked(staged):
    results, gguf = staged
    artifact = load_artifacts(["m"], gguf)[0]
    handle = artifact.handle("ryzewtf")
    assert not already_published(artifact, handle, artifact.sha256)

    (results / "publish_m.json").write_text(
        json.dumps({"handle": handle, "sha256": artifact.sha256}), encoding="utf-8")
    assert already_published(artifact, handle, artifact.sha256)
    # A re-converted artifact under the same handle is a different file and must go up again.
    assert not already_published(artifact, handle, "ff" * 32)


def test_the_handle_lands_in_models_yaml_next_to_the_hash(tmp_path, monkeypatch):
    import scripts.kaggle_publish as kp

    models = tmp_path / "models.yaml"
    models.write_text(
        "schema_version: 1\n\nmodels:\n\n  olmoe-0924:\n    hf_repo: allenai/OLMoE-1B-7B-0924\n"
        "    gguf: {repo: null, file: null, sha256: null, size_bytes: null}   # T1.1\n",
        encoding="utf-8")
    monkeypatch.setattr(kp, "MODELS_CONFIG", models)

    artifact = Artifact(model_key="olmoe-0924", slug_suffix="olmoe-0924",
                        path=tmp_path / "olmoe-0924-Q4_K_M.gguf", sha256="ab" * 32,
                        size_bytes=42, llama_commit="c" * 40, transformers="4.57.6")
    changes = register(artifact, "ryzewtf/moe-panel/gguf/olmoe-0924", "ab" * 32, force=True)

    assert changes
    text = models.read_text(encoding="utf-8")
    assert "kaggle: ryzewtf/moe-panel/gguf/olmoe-0924" in text
    import yaml
    block = yaml.safe_load(text)["models"]["olmoe-0924"]["gguf"]
    assert block["repo"] is None and block["sha256"] == "ab" * 32


def test_an_f16_extra_is_not_written_into_the_model_block():
    """models.yaml has one gguf: per model, and a second precision is not a second model."""
    artifact = Artifact(model_key="olmoe-0125", slug_suffix="olmoe-0125-f16",
                        path=__import__("pathlib").Path("olmoe-0125-F16.gguf"), sha256="ab" * 32,
                        size_bytes=1, llama_commit="c" * 40, transformers="4.57.6", is_f16=True)
    assert register(artifact, "ryzewtf/moe-panel/gguf/olmoe-0125-f16", "ab" * 32,
                    force=True) == []


def test_version_notes_carry_the_provenance_a_dataset_page_cannot_otherwise_show():
    artifact = Artifact(model_key="m", slug_suffix="m",
                        path=__import__("pathlib").Path("m-Q4_K_M.gguf"), sha256="ab" * 32,
                        size_bytes=1, llama_commit="7077abbe14c510cb", transformers="5.15.1")
    notes = artifact.version_notes()
    assert "7077abbe14c5" in notes and "transformers 5.15.1" in notes and "ab" * 32 in notes


def test_the_handle_is_a_kaggle_model_variation_not_a_dataset_slug():
    """Four parts. Kaggle Models bills against a private quota separate from Datasets', which is
    the whole reason the panel lives here: the dataset quota stays free for corpora and traces."""
    artifact = Artifact(model_key="qwen3-30b-a3b", slug_suffix="qwen3-30b-a3b",
                        path=__import__("pathlib").Path("qwen3-30b-a3b-Q4_K_M.gguf"),
                        sha256="ab" * 32, size_bytes=1, llama_commit="c" * 40,
                        transformers="4.57.6")
    handle = artifact.handle("ryzewtf")
    assert handle == "ryzewtf/moe-panel/gguf/qwen3-30b-a3b"

    from kagglehub.handle import parse_model_handle
    parsed = parse_model_handle(handle)
    assert (parsed.owner, parsed.model, parsed.variation) == (
        "ryzewtf", "moe-panel", "qwen3-30b-a3b")
    assert parsed.framework_enum().name.endswith("GGUF")


def test_version_notes_name_the_upstream_licence():
    """No license_name is sent to Kaggle -- these are derived weights and the licence is the
    original publisher's to assert. Naming it in the notes is the honest middle."""
    artifact = Artifact(model_key="gemma-4-26b-a4b", slug_suffix="gemma-4-26b-a4b",
                        path=__import__("pathlib").Path("gemma-4-26b-a4b-Q4_K_M.gguf"),
                        sha256="ab" * 32, size_bytes=1, llama_commit="c" * 40,
                        transformers="5.15.1")
    assert "upstream licence Gemma Terms of Use" in artifact.version_notes()


# -- the other end: resolving a published handle on Kaggle ------------------------------------


def _fake_kagglehub(monkeypatch, root):
    """Inject a kagglehub whose model_download returns `root`, recording the handle asked for."""
    import sys
    import types

    calls = []
    module = types.ModuleType("kagglehub")
    module.model_download = lambda handle, *a, **k: (calls.append(handle), str(root))[1]
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    return calls


def _setup_ctx(tmp_path):
    from src.runtime.setup_kaggle import SetupContext

    return SetupContext(scratch=tmp_path / "scratch", dry_run=False, jobs=1, cuda_arch="75",
                        llama_commit="c" * 40, quant="Q4_K_M", models=(), hf_token_present=False)


def test_a_published_handle_is_how_a_collection_session_finds_a_converted_gguf(
    tmp_path, monkeypatch
):
    """`repo: null` says "this came from no HF repo"; `kaggle:` says where it went instead."""
    import scripts.kaggle_setup as ks

    published = tmp_path / "downloaded"
    published.mkdir()
    (published / "olmoe-0924-Q4_K_M.gguf").write_bytes(b"0123456789")
    calls = _fake_kagglehub(monkeypatch, published)
    monkeypatch.setattr(ks, "find_attached_gguf", lambda name: None)
    monkeypatch.setattr(ks, "RESULTS", tmp_path / "results")

    meta = {"gguf": {"repo": None, "kaggle": "ryzewtf/moe-panel/gguf/olmoe-0924",
                     "file": "olmoe-0924-Q4_K_M.gguf", "size_bytes": 10, "sha256": None}}
    path = ks.acquire_gguf("olmoe-0924", meta, _setup_ctx(tmp_path), verify_hash=False)

    assert path == published / "olmoe-0924-Q4_K_M.gguf"
    assert calls == ["ryzewtf/moe-panel/gguf/olmoe-0924"]


def test_a_copy_this_session_just_converted_beats_re_downloading_it(tmp_path, monkeypatch):
    import scripts.kaggle_setup as ks

    ctx = _setup_ctx(tmp_path)
    (ctx.scratch / "gguf").mkdir(parents=True)
    local = ctx.scratch / "gguf" / "olmoe-0924-Q4_K_M.gguf"
    local.write_bytes(b"0123456789")
    calls = _fake_kagglehub(monkeypatch, tmp_path / "unused")
    monkeypatch.setattr(ks, "find_attached_gguf", lambda name: None)
    monkeypatch.setattr(ks, "RESULTS", tmp_path / "results")

    meta = {"gguf": {"repo": None, "kaggle": "ryzewtf/moe-panel/gguf/olmoe-0924",
                     "file": "olmoe-0924-Q4_K_M.gguf", "size_bytes": 10, "sha256": None}}
    assert ks.acquire_gguf("olmoe-0924", meta, ctx, verify_hash=False) == local
    assert calls == []


def test_a_dataset_that_does_not_hold_the_named_file_is_a_stop(tmp_path, monkeypatch):
    import scripts.kaggle_setup as ks

    empty = tmp_path / "downloaded"
    empty.mkdir()
    _fake_kagglehub(monkeypatch, empty)
    monkeypatch.setattr(ks, "find_attached_gguf", lambda name: None)

    meta = {"gguf": {"repo": None, "kaggle": "ryzewtf/moe-panel/gguf/olmoe-0924",
                     "file": "olmoe-0924-Q4_K_M.gguf", "size_bytes": 10, "sha256": None}}
    with pytest.raises(ks.SetupError, match="disagree about what was published"):
        ks.acquire_gguf("olmoe-0924", meta, _setup_ctx(tmp_path), verify_hash=False)
