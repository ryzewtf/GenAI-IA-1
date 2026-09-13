"""Pinned run configuration and its hash — plan S.3.

Every numerics-visible flag lives in ``configs/run.yaml`` under ``hashed:``. This module
computes ``run_config_sha256`` over the canonical serialization of that block and writes it
into every shard manifest.

Why this is load-bearing
------------------------
Document-level sharding is bit-exact *only* because each document is prefilled with a cleared
KV cache and no cross-document state. Concatenating shards in document order reproduces a
single-run trace byte for byte — provided every run flag, the build commit, and the GPU
architecture are identical across the sessions that produced them. A shard collected under a
different config is a different experiment, so :func:`assert_shards_compatible` refuses to
merge rather than warning.

Operational settings (upload retries, scratch paths, session limits) live under ``unhashed:``
and deliberately do *not* invalidate collected shards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared in requirements.lock
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

__all__ = [
    "RunConfig",
    "ConfigError",
    "IncompatibleShardError",
    "canonical_hash",
    "assert_shards_compatible",
]

DEFAULT_RUN_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run.yaml"


class ConfigError(RuntimeError):
    """The run configuration is missing, malformed, or not ready for collection."""


class IncompatibleShardError(RuntimeError):
    """Shards were collected under different conditions and must not be merged."""


def canonical_hash(payload: Mapping[str, Any]) -> str:
    """SHA256 over a canonical JSON rendering of ``payload``.

    Sorted keys and no insignificant whitespace, so the hash depends on the *values* and not
    on YAML formatting or dict ordering. ``allow_nan=False`` because a NaN in a pinned config
    would produce a hash that compares unequal to itself.
    """
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    """Loaded ``run.yaml``. ``sha256`` covers the ``hashed`` block only."""

    hashed: dict[str, Any]
    unhashed: dict[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "RunConfig":
        path = Path(path) if path is not None else DEFAULT_RUN_CONFIG
        if not path.exists():
            raise ConfigError(f"{path}: run config not found")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        if "hashed" not in raw:
            raise ConfigError(
                f"{path}: missing the `hashed:` block. Every numerics-visible flag must live "
                "there so it is covered by run_config_sha256."
            )
        return cls(
            hashed=raw["hashed"],
            unhashed=raw.get("unhashed", {}) or {},
            source_path=path,
        )

    # -- identity -----------------------------------------------------------------------

    @property
    def sha256(self) -> str:
        return canonical_hash(self.hashed)

    @property
    def short(self) -> str:
        return self.sha256[:12]

    # -- typed accessors ----------------------------------------------------------------

    @property
    def platform(self) -> dict[str, Any]:
        return self.hashed["platform"]

    @property
    def build(self) -> dict[str, Any]:
        return self.hashed["build"]

    @property
    def inference(self) -> dict[str, Any]:
        return self.hashed["inference"]

    @property
    def capture(self) -> dict[str, Any]:
        return self.hashed["capture"]

    @property
    def analysis(self) -> dict[str, Any]:
        return self.hashed["analysis"]

    @property
    def gpu_arch(self) -> int:
        return int(self.platform["gpu_arch"])

    @property
    def engine(self) -> str:
        """Collection engine. Absent == ``llama_cpp`` (run.yaml predates the vLLM port)."""
        return str(self.build.get("engine", "llama_cpp"))

    @property
    def epsilon_mix(self) -> float:
        return float(self.analysis["epsilon_mix"])

    # -- readiness ----------------------------------------------------------------------

    def assert_collection_ready(self) -> None:
        """Refuse to start a capture run under an under-specified config.

        The build/inference knobs that are numerics-visible differ by engine, so this dispatches
        on ``build.engine``. The capture and analysis invariants are engine-independent (the trace
        files and how the analysis is defined do not change with the engine) and are checked for
        both. Everything here is cheap to get wrong and expensive to discover after a paid session.
        """
        problems: list[str] = []
        if self.engine == "vllm":
            self._collect_vllm_problems(problems)
        elif self.engine == "llama_cpp":
            self._collect_llama_cpp_problems(problems)
        else:
            problems.append(
                f"build.engine={self.engine!r} is unknown — expected 'vllm' or 'llama_cpp'"
            )
        self._collect_capture_analysis_problems(problems)

        if problems:
            raise ConfigError(
                f"{self.source_path} is not ready for collection:\n  - "
                + "\n  - ".join(problems)
            )

    def _collect_llama_cpp_problems(self, problems: list[str]) -> None:
        """llama.cpp-specific readiness: build commit, context cap, tensor split, flash-attn."""
        if not self.build.get("llama_cpp_commit"):
            problems.append(
                "build.llama_cpp_commit is null — pin it from T0.2 before collecting"
            )
        if self.build.get("ggml_native", False):
            problems.append(
                "build.ggml_native must be false — a binary built with -march=native SIGILLs "
                "intermittently on a different Kaggle host"
            )

        if not self.inference.get("ctx_size"):
            problems.append("inference.ctx_size must be set explicitly (invariant I4)")

        if self.inference.get("n_gpu_layers", 0) < 1:
            problems.append("inference.n_gpu_layers < 1 — CPU layers are strictly worse here")

        n_devices = int(self.platform.get("n_devices", 1))
        split = self.inference.get("tensor_split")
        if n_devices > 1 and not split:
            problems.append(
                "inference.tensor_split must be pinned explicitly on a multi-GPU run — an "
                "auto split varies with free VRAM and breaks shard-merge exactness"
            )
        if split is not None and len(split) != n_devices:
            problems.append(
                f"inference.tensor_split has {len(split)} entries but platform.n_devices is "
                f"{n_devices}"
            )

        if self.inference.get("flash_attn") is None:
            problems.append(
                "inference.flash_attn must be pinned true or false — an unpinned default is a "
                "silent confound across sessions"
            )
        if self.inference.get("override_tensor"):
            problems.append(
                "inference.override_tensor must be null at collection time; test the "
                "CPU-pinned router once in T3.7 instead"
            )

    def _collect_vllm_problems(self, problems: list[str]) -> None:
        """vLLM-specific readiness: the Turing-forced knobs and a pinned context cap.

        On sm_75 three settings are not optional and each is numerics-visible: the V0 engine (V1
        refuses compute capability < 8.0), eager mode (Turing has no usable CUDA-graph MoE path,
        and it is also the precondition for the capture hooks), and fp16 (bf16 is unsupported). A
        config that leaves any of them to a default is a silent confound across sessions.
        """
        if not self.build.get("vllm_version"):
            problems.append(
                "build.vllm_version is null — pin it (0.10.2 is the P0/P1-verified pin)"
            )
        if not self.build.get("attention_backend"):
            problems.append(
                "build.attention_backend must be pinned — vLLM selects XFormers on Turing "
                "(FA2 is unavailable), and an unpinned default is a silent confound"
            )
        if not self.build.get("enforce_eager", False):
            problems.append(
                "build.enforce_eager must be true — Turing has no usable CUDA-graph MoE path and "
                "the capture hooks require eager execution"
            )

        turing = self.gpu_arch < 80
        if turing and str(self.build.get("engine_version", "")).lower() != "v0":
            problems.append(
                "build.engine_version must be 'v0' on compute capability < 8.0 — vLLM V1 "
                "hard-raises NotImplementedError there (set VLLM_USE_V1=0)"
            )
        dtype = str(self.build.get("dtype", "")).lower()
        if not dtype:
            problems.append("build.dtype must be set explicitly (float16 on Turing)")
        elif turing and dtype in ("bfloat16", "bf16"):
            problems.append(
                f"build.dtype={dtype!r} is unsupported on Turing (sm_75) — use float16"
            )

        if not self.inference.get("max_model_len"):
            problems.append(
                "inference.max_model_len must be set explicitly (invariant I4; never the arch "
                "default — some panel models default to a 262144-token context)"
            )

    def _collect_capture_analysis_problems(self, problems: list[str]) -> None:
        """Engine-independent invariants: prefill-only capture, cleared KV, topk, and finite MI."""
        if self.capture.get("mode") != "prefill_only":
            problems.append("capture.mode must be 'prefill_only'")
        if not self.capture.get("clear_kv_between_docs", False):
            problems.append(
                "capture.clear_kv_between_docs must be true — it is what makes document-level "
                "sharding bit-exact"
            )
        if "topk" not in (self.capture.get("streams") or []):
            problems.append("capture.streams must include 'topk' — it is the labels (I1)")

        if self.analysis.get("clamp_mi_at_zero", False):
            problems.append(
                "analysis.clamp_mi_at_zero must be false — a negative I-hat is diagnostic "
                "information about the probe (plan §1.2)"
            )
        eps = self.analysis.get("epsilon_mix")
        if not eps or eps <= 0:
            problems.append("analysis.epsilon_mix must be a positive float")

    # -- manifest glue ------------------------------------------------------------------

    def engine_build(self) -> str:
        """Engine-neutral build identity for the manifest — ``<engine>@<version>``.

        The manifest's ``engine_build`` (a required, merge-invariant key) records which engine and
        version produced the shard. llama.cpp uses the pinned commit; vLLM the pinned version. A
        shard whose engine_build differs is a different experiment (plan S.3).
        """
        if self.engine == "vllm":
            return f"vllm@{self.build.get('vllm_version')}"
        return f"llama_cpp@{self.build.get('llama_cpp_commit')}"

    def manifest_fields(self) -> dict[str, Any]:
        """The subset of manifest fields this config determines."""
        return {
            "run_config_sha256": self.sha256,
            "engine_build": self.engine_build(),
            # Kept for llama.cpp readers; null for vLLM (its build identity is engine_build).
            "llama_cpp_commit": self.build.get("llama_cpp_commit"),
            "hidden_subsample_n": self.capture.get("hidden_subsample_n"),
            "device_plan": {
                "n_gpu": self.platform.get("n_devices"),
                "split_mode": self.inference.get("split_mode"),
                "tensor_split": self.inference.get("tensor_split"),
                "gpu_arch": self.gpu_arch,
            },
            "capture_flags": {
                "pre_topk": True,
                "pre_norm": True,
                "topk_captured": "topk" in (self.capture.get("streams") or []),
            },
        }


def assert_shards_compatible(
    manifests: Iterable[Mapping[str, Any]],
    invariant_keys: Iterable[str],
) -> None:
    """Hard-fail if any shard disagrees with the first on a merge-invariant key.

    Plan S.3: "A shard collected under a different config is a different experiment; refuse
    to merge it." This is the enforcement point.
    """
    manifests = list(manifests)
    if not manifests:
        raise IncompatibleShardError("no shards found")

    reference = manifests[0]
    ref_id = reference.get("shard_id", "?")
    conflicts: list[str] = []

    for manifest in manifests[1:]:
        shard_id = manifest.get("shard_id", "?")
        for key in invariant_keys:
            want, got = reference.get(key), manifest.get(key)
            if want != got:
                conflicts.append(
                    f"  {key}: shard {ref_id} has {want!r}, shard {shard_id} has {got!r}"
                )

    if conflicts:
        raise IncompatibleShardError(
            "refusing to merge shards collected under different conditions "
            "(plan S.3) — these are different experiments:\n" + "\n".join(conflicts)
        )
