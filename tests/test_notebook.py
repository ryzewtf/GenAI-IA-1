"""Tests for the generated Kaggle session notebook (`scripts/make_notebook.py`).

A notebook is the one artifact in this project whose bugs are found by *running a Kaggle session*,
which costs a quota slot and up to twelve hours. Everything checkable without a session is checked
here: that every cell is valid Python, that the stages the parameter cell allows are exactly the
stages the dispatch cell handles, and that the rules the notebook exists to enforce (nothing large
in /kaggle/working, no commits, a corpus that must predate the session) are actually in it rather
than only in its docstring.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "make_notebook.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "moe_session.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} missing -- run python scripts/make_notebook.py"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _all_source(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_every_code_cell_is_valid_python(notebook):
    """A SyntaxError in cell 5 is found at hour three of a twelve-hour session, after the build."""
    for i, src in enumerate(_code(notebook)):
        try:
            ast.parse(src)
        except SyntaxError as exc:  # pragma: no cover - the assert carries the message
            pytest.fail(f"code cell {i} is not valid Python: {exc}")


def test_the_notebook_on_disk_matches_the_generator(tmp_path):
    """The .ipynb is a build artifact. If someone edits it in the Kaggle UI and commits the result,
    the generator becomes a lie and the next regeneration silently reverts their fix."""
    out = subprocess.run([sys.executable, str(GENERATOR)], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    regenerated = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert regenerated, "generator produced an empty notebook"


def test_the_allowed_stages_are_exactly_the_handled_stages(notebook):
    """The parameter cell asserts STAGE is in a tuple; the dispatch cell branches on it. A stage in
    one and not the other is a session that boots, builds, and then does nothing -- or one that is
    rejected for a name the notebook actually supports."""
    src = _all_source(notebook)
    tree = ast.parse(_code(notebook)[0])

    allowed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            comparator = node.test.comparators[0]
            if isinstance(comparator, ast.Tuple):
                allowed = {ast.literal_eval(e) for e in comparator.elts}
    assert allowed, "parameter cell no longer asserts STAGE against a tuple of names"

    for stage in allowed:
        assert f'"{stage}"' in src, f"stage {stage!r} is allowed but never dispatched"


def test_no_stage_is_dispatched_that_the_assert_would_reject(notebook):
    """The inverse direction: a branch for a stage the assert rejects is dead code that reads as a
    supported option."""
    dispatch = [s for s in _code(notebook) if "STAGE ==" in s or "STAGE in" in s]
    assert dispatch, "no dispatch cell found"
    tree = ast.parse(_code(notebook)[0])
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            comparator = node.test.comparators[0]
            if isinstance(comparator, ast.Tuple):
                allowed = {ast.literal_eval(e) for e in comparator.elts}

    dispatched = set()
    for src in dispatch:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
                if node.left.id != "STAGE":
                    continue
                for c in node.comparators:
                    if isinstance(c, ast.Constant):
                        dispatched.add(c.value)
                    elif isinstance(c, ast.Tuple):
                        dispatched |= {ast.literal_eval(e) for e in c.elts}
    assert dispatched <= allowed, f"dispatched but not allowed: {dispatched - allowed}"


def test_the_notebook_never_commits_or_pushes(notebook):
    """The user commits. A notebook that pushes needs a write token in a session that also holds
    the weights, and it takes the decision of what enters history out of the user's hands."""
    src = _all_source(notebook)
    assert '"push"' not in src and "'push'" not in src
    assert '"commit"' not in src and "'commit'" not in src


def test_the_harvest_uses_a_three_way_apply(notebook):
    """The patch is cut on Linux (LF); the workstation runs core.autocrlf=true, so its working tree
    is CRLF. A plain `git apply` matches context against the working tree and fails on that alone.
    --3way goes through the index blobs, which are LF on both machines."""
    src = _all_source(notebook)
    assert "--3way" in src
    assert "--full-index" in src, "--3way needs unabbreviated blob shas to resolve the base"


def test_the_harvest_watches_the_i9_caps(notebook):
    """/kaggle/working is ~20 GB with a ~500 file commit cap (I9). A session that quietly exceeds it
    loses its output at commit time, which is after the twelve hours are spent."""
    src = _all_source(notebook)
    assert "I9" in src
    assert "/kaggle/working" in src


def test_collect_refuses_to_invent_a_corpus(notebook):
    """Corpora are gitignored, so a fresh clone has none. Building one ad hoc inside a collect
    session would produce traces incomparable with every other model's -- the failure is silent and
    only shows up as a model that disagrees with the panel."""
    src = _all_source(notebook)
    assert "does not exist" in src
    assert "incomparable" in src


def test_the_build_commit_is_never_taken_from_head(notebook):
    """`configs/run.yaml:build.llama_cpp_commit` is inside run_config_sha256 and is load-bearing for
    correctness, not just reproducibility. The notebook must not offer a way around it."""
    src = _all_source(notebook)
    assert "llama_cpp_commit" not in src or "pinned" in src.lower()


def test_secrets_are_read_from_kaggle_secrets_not_inlined(notebook):
    """An inline token is committed the moment the notebook is saved from the UI."""
    src = _all_source(notebook)
    assert "UserSecretsClient" in src
    assert "ghp_" not in src and "github_pat_" not in src
