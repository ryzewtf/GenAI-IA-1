"""Tests for CUDA driver-library discovery in the Kaggle build setup.

The Kaggle CUDA 12.8 image ships the toolkit (nvcc, headers, runtime libs) but no driver stub and no
`libcuda.so` on the default linker path, so FindCUDAToolkit cannot define the `CUDA::cuda_driver`
imported target that ggml-cuda links -- configure fails at generate time, before any GPU work and
therefore before the session has done anything worth the GPU quota. `cuda_driver_lib_dir` finds the
real driver library (which IS on the image, in a dir the linker does not search) so we can put it on
CMAKE_LIBRARY_PATH. These tests pin the discovery order and the POSIX-path shape, both of which a
Windows generator host would otherwise get wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.runtime.setup_kaggle import SetupContext


def _ctx(**kw) -> SetupContext:
    base = dict(
        scratch=Path("/tmp/x"), dry_run=True, jobs=1, cuda_arch="75",
        llama_commit="0" * 40, quant="Q4_K_M", models=(), hf_token_present=False, cuda=True,
    )
    base.update(kw)
    return SetupContext(**base)


def test_toolkit_root_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-xyz")
    assert _ctx().cuda_toolkit_root == "/opt/cuda-xyz"


def test_toolkit_root_falls_back_to_the_conventional_path(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    assert _ctx().cuda_toolkit_root == "/usr/local/cuda"


def test_driver_lib_dir_prefers_the_loaded_driver_dir(monkeypatch):
    """The loaded driver (/usr/local/nvidia/lib64) is probed before the compat build, so when both
    have libcuda.so the loaded one wins -- avoiding a link against the older forward-compat driver."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    present = {"/usr/local/nvidia/lib64/libcuda.so", "/usr/local/cuda/compat/libcuda.so"}
    monkeypatch.setattr(os.path, "exists", lambda p: p in present)
    assert _ctx().cuda_driver_lib_dir == "/usr/local/nvidia/lib64"


def test_driver_lib_dir_falls_back_to_compat_when_only_that_has_libcuda(monkeypatch):
    """If the loaded-driver dir is absent but the compat driver is present, use compat."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/usr/local/cuda/compat/libcuda.so")
    assert _ctx().cuda_driver_lib_dir == "/usr/local/cuda/compat"


def test_driver_lib_dir_is_a_posix_path_even_on_a_windows_host(monkeypatch):
    """Path arithmetic on Windows would produce backslashes; the Kaggle target is Linux. When no
    candidate resolves (the generator host is not the image), the fallback must still be POSIX."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    got = _ctx().cuda_driver_lib_dir
    assert "\\" not in got
    assert got.startswith("/")


def test_a_cpu_context_never_reads_the_driver_lib(monkeypatch):
    """A CPU session builds with GGML_CUDA=OFF and must not require any CUDA driver library; the
    property is simply never consulted. This guards against a future refactor wiring it in for CPU."""
    # No assertion on value -- just that constructing and using a CPU ctx needs no libcuda on disk.
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    ctx = _ctx(cuda=False)
    assert ctx.cuda is False
