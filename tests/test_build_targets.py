"""Which llama.cpp targets each Kaggle session builds.

`cmake --build --target X` builds only the named targets, so a tool a later step invokes that is not
named here silently does not exist -- and the gap surfaces only after the slow, GPU-metered build.
That failure has cost a paid session twice (CPU path missing llama-eval-callback; then the CUDA path
building moe_trace but not llama-eval-callback and failing T1.4 after ~30 min of build). These tests
pin the full matrix so a third time is a red test, not a lost session.
"""

from __future__ import annotations

from scripts.kaggle_setup import _build_targets


def test_cuda_collect_with_gates_builds_the_collector_and_the_scan_tool():
    """The default collect session: CUDA + gates. It runs moe_trace AND the T1.4 node scan, so both
    moe_trace and llama-eval-callback must be built. This is the exact case that failed."""
    assert _build_targets(cuda=True, skip_gates=False) == ("moe_trace", "llama-eval-callback")


def test_cuda_with_skip_gates_builds_only_the_collector():
    assert _build_targets(cuda=True, skip_gates=True) == ("moe_trace",)


def test_cpu_gates_builds_the_scan_tool_and_quantize_never_moe_trace():
    """A CPU session runs the node scan and may requantize; moe_trace needs a GPU it does not have,
    so building it would only burn minutes."""
    targets = _build_targets(cuda=False, skip_gates=False)
    assert "llama-eval-callback" in targets
    assert "llama-quantize" in targets
    assert "moe_trace" not in targets


def test_cpu_target_set_does_not_depend_on_skip_gates():
    """The CPU set is fixed: llama-quantize is for conversion, and llama-eval-callback is cheap and
    harmless to have built even when a particular CPU run skips gates."""
    assert _build_targets(cuda=False, skip_gates=True) == _build_targets(cuda=False, skip_gates=False)


def test_every_session_builds_the_tool_its_own_gates_step_needs():
    """The invariant behind the two lost sessions: if gates run (not skipped), llama-eval-callback
    is built, on both accelerators."""
    for cuda in (True, False):
        assert "llama-eval-callback" in _build_targets(cuda=cuda, skip_gates=False)
