"""Tests for the pinned llama.cpp patch mechanism (`build.llama_cpp_patches`).

The mechanism exists for one patch — `patches/gemma4-router-input-cb.patch`, which names the tensor
Gemma 4's router actually consumes — but the properties it has to hold are general, and every one of
them is a way a session could quietly build the wrong binary:

* the patch bytes are what `run.yaml` says they are, so `run_config_sha256` describes the tree;
* applying is idempotent, because `step_llama` is re-entered by every resumed session;
* anything unexpected is fatal, because the alternative is a binary missing a capture node and a
  failure that surfaces only after a setup's worth of session time is spent.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.runtime.setup_kaggle import (
    REPO_ROOT,
    SetupContext,
    SetupError,
    _apply_patches,
    _read_patch_pins,
    step_llama,
)

PATCH_REL = "patches/gemma4-router-input-cb.patch"
PATCH_PATH = REPO_ROOT / PATCH_REL
TARGET = "src/models/gemma4.cpp"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ctx(tmp_path: Path, pins) -> SetupContext:
    return SetupContext(
        scratch=tmp_path, dry_run=False, jobs=1, cuda_arch="75",
        llama_commit="0" * 40, llama_patches=tuple(pins), quant="Q4_K_M",
        models=(), hf_token_present=False,
    )


@pytest.fixture
def fake_llama(tmp_path: Path) -> Path:
    """A git repo holding just the file the patch touches, with the pre-patch lines around it.

    A real llama.cpp clone is ~200 MB and a network round trip; the patch only needs its context
    lines to match, so the smallest honest fixture is the hunk's own context.
    """
    root = tmp_path / "llama.cpp"
    (root / "src" / "models").mkdir(parents=True)
    src = root / "src" / "models" / "gemma4.cpp"
    body = PATCH_PATH.read_text(encoding="utf-8")
    context = [ln[1:] for ln in body.splitlines()
               if ln.startswith((" ", "-")) and not ln.startswith("---")]
    # The hunk starts at line 290 upstream; pad so the offsets are realistic rather than exact.
    src.write_text("\n".join(["// filler"] * 289 + context) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "base"], cwd=root, check=True)
    return root


def test_the_shipped_patch_matches_the_pin_in_run_yaml():
    """The pin is inside run_config_sha256. If the file and the pin disagree, every manifest written
    since the divergence is wrong about what produced it."""
    import yaml

    run_yaml = yaml.safe_load((REPO_ROOT / "configs" / "run.yaml").read_text(encoding="utf-8"))
    pins = run_yaml["hashed"]["build"]["llama_cpp_patches"]
    assert len(pins) == 1
    assert pins[0]["file"] == PATCH_REL
    assert pins[0]["sha256"] == _sha(PATCH_PATH)


def test_the_patch_is_covered_by_run_config_sha256():
    """Not a tautology: the block could have been added under `unhashed:`, where it would document
    the build without binding it."""
    from src.runtime.config import RunConfig

    cfg = RunConfig.load(REPO_ROOT / "configs" / "run.yaml")
    assert "llama_cpp_patches" in cfg.hashed["build"]
    assert "llama_cpp_patches" not in (cfg.unhashed.get("build") or {})


def test_the_patch_adds_a_callback_and_nothing_else():
    """A callback names an existing tensor; it creates no op and changes no arithmetic. That is the
    entire argument for why patching off the pinned commit is safe for the other six models, so it
    is worth pinning as a test rather than leaving in a comment."""
    added = [ln for ln in PATCH_PATH.read_text(encoding="utf-8").splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln for ln in PATCH_PATH.read_text(encoding="utf-8").splitlines()
               if ln.startswith("-") and not ln.startswith("---")]
    assert removed == [], f"the patch removes lines: {removed}"
    assert len(added) == 1, f"expected exactly one added line, got {added}"
    assert 'cb(tmp, "ffn_moe_router_input", il);' in added[0]


def test_applying_is_idempotent(fake_llama, tmp_path):
    """`step_llama` is re-entered by every resumed session, so the second call must be a skip and
    not a failure that has to be interpreted at hour eleven."""
    ctx = _ctx(tmp_path, [(PATCH_REL, _sha(PATCH_PATH))])
    first = _apply_patches(ctx)
    assert [r["status"] for r in first] == ["applied"]
    second = _apply_patches(ctx)
    assert [r["status"] for r in second] == ["already-applied"]
    assert "ffn_moe_router_input" in (fake_llama / TARGET).read_text(encoding="utf-8")


def test_a_patch_whose_bytes_changed_is_refused(fake_llama, tmp_path):
    """The failure this exists for: someone edits the patch, the binary changes, and the recorded
    run config hash keeps insisting nothing moved."""
    ctx = _ctx(tmp_path, [(PATCH_REL, "0" * 64)])
    with pytest.raises(SetupError, match="hashes .* but run.yaml pins"):
        _apply_patches(ctx)
    assert "ffn_moe_router_input" not in (fake_llama / TARGET).read_text(encoding="utf-8")


def test_a_missing_patch_file_is_fatal(tmp_path):
    ctx = _ctx(tmp_path, [("patches/does-not-exist.patch", "0" * 64)])
    with pytest.raises(SetupError, match="does not exist"):
        _apply_patches(ctx)


def test_a_patch_that_does_not_apply_is_fatal_not_a_warning(tmp_path):
    """Warning and building anyway produces a binary missing a capture node, and the first symptom
    is a node the harness cannot find after the setup time is already spent."""
    root = tmp_path / "llama.cpp"
    (root / "src" / "models").mkdir(parents=True)
    (root / "src" / "models" / "gemma4.cpp").write_text("something else entirely\n",
                                                        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "base"], cwd=root, check=True)

    ctx = _ctx(tmp_path, [(PATCH_REL, _sha(PATCH_PATH))])
    with pytest.raises(SetupError, match="neither applies to nor is already applied"):
        _apply_patches(ctx)


def test_no_patches_configured_is_a_clean_no_op(tmp_path):
    assert _apply_patches(_ctx(tmp_path, [])) == []


def test_a_pin_without_a_hash_is_rejected():
    """Silently ignoring a malformed entry is exactly the failure the mechanism exists to prevent."""
    with pytest.raises(SetupError, match="must be a mapping"):
        _read_patch_pins({"llama_cpp_patches": [{"file": PATCH_REL}]})
    with pytest.raises(SetupError, match="must be a mapping"):
        _read_patch_pins({"llama_cpp_patches": ["patches/x.patch"]})


def test_absent_patches_key_means_none():
    assert _read_patch_pins({}) == ()
    assert _read_patch_pins({"llama_cpp_patches": None}) == ()


def test_step_llama_applies_patches_when_the_tree_is_already_at_the_commit(fake_llama, tmp_path):
    """The multi-model regression: a prior model with no patches of its own leaves the tree at the
    pinned commit, so the next model hits the "already at commit" branch. That branch used to return
    before applying patches, so Gemma built without its router-input callback and T1.4 halted. The
    skip path must apply patches too -- it is idempotent, so this is safe."""
    # step_llama looks for scratch/llama.cpp; the fake_llama fixture already lives at
    # tmp_path/llama.cpp, so scratch=tmp_path makes ctx.llama_dir resolve to it directly.
    assert fake_llama == tmp_path / "llama.cpp"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fake_llama,
                          capture_output=True, text=True, check=True).stdout.strip()
    ctx = SetupContext(
        scratch=tmp_path, dry_run=False, jobs=1, cuda_arch="75",
        llama_commit=head, llama_patches=((PATCH_REL, _sha(PATCH_PATH)),), quant="Q4_K_M",
        models=(), hf_token_present=False,
    )
    result = step_llama(ctx)
    assert result.status == "skipped"
    assert [r["status"] for r in result.data["patches"]] == ["applied"]
    assert "ffn_moe_router_input" in (fake_llama / TARGET).read_text(encoding="utf-8")


def test_gemma_router_input_points_at_the_patched_node():
    """models.yaml and the patch have to agree on the name, and nothing else in the panel may pick
    it up -- the other six architectures pass the router the same tensor they pass the experts."""
    import yaml

    models = yaml.safe_load(
        (REPO_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    gemma = models["gemma-4-26b-a4b"]["node_names"]
    assert gemma["router_input"] == "ffn_moe_router_input-%d"

    for key, model in models.items():
        if key == "gemma-4-26b-a4b":
            continue
        assert model["node_names"]["router_input"] != "ffn_moe_router_input-%d", key
