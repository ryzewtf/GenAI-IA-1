"""Strip the vLLM-incompatible ``scale_dtype`` / ``zp_dtype`` keys from a compressed-tensors model.

compressed-tensors 0.16 (bundled by llmcompressor 0.11.0) stamps two newer, currently-``None`` fields
into ``config.json``'s ``quantization_config``: ``scale_dtype`` and ``zp_dtype`` (PR #508). Released
vLLM (incl. the 0.10.2 the trace campaign is PINNED to via ``engine_build`` / ``run_config_sha256``)
rejects them at engine-config parse::

    ValidationError: 2 validation errors for VllmConfig
      scale_dtype  Extra inputs are not permitted [extra_forbidden, input_value=None]
      zp_dtype     Extra inputs are not permitted [extra_forbidden, input_value=None]

See llm-compressor#2057 (Red Hat's remedy is exactly this removal). The fields are ``None`` and carry
no quantization information — the real quant data is ``config_groups`` + ``weight_packed`` /
``weight_scale`` — so dropping them is lossless and does NOT require re-quantizing. We cannot bump vLLM
instead: its version is baked into the vLLM panel's ``run_config_sha256`` and a different engine build
would orphan the other models' traces.

Two modes:
  * ``--repo-id R``  : download ``config.json`` from the HF repo, strip, re-upload just that file
                       (needs HF_TOKEN in env; the ~2h quant is NOT re-run). Idempotent.
  * ``--local DIR``  : strip ``DIR/config.json`` in place.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

#: The exact keys released vLLM's VllmConfig refuses (llm-compressor#2057). Removed only when null.
INCOMPATIBLE_KEYS = ("scale_dtype", "zp_dtype")


def strip_incompatible_keys(config: dict[str, Any]) -> list[str]:
    """Remove INCOMPATIBLE_KEYS anywhere inside ``quantization_config`` (recursively). In place.

    Returns the dotted paths removed, so the caller can report exactly what changed (empty = the
    config was already clean, which is a valid no-op, not an error).
    """
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return []
    removed: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key in INCOMPATIBLE_KEYS:
                if key in node:
                    node.pop(key)
                    removed.append(f"{path}.{key}" if path else key)
            for k, v in list(node.items()):
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")

    _walk(qc, "quantization_config")
    return removed


def _patch_text(config_text: str) -> tuple[str, list[str]]:
    config = json.loads(config_text)
    removed = strip_incompatible_keys(config)
    return json.dumps(config, indent=2) + "\n", removed


def patch_local(config_path: Path) -> list[str]:
    new_text, removed = _patch_text(config_path.read_text(encoding="utf-8"))
    if removed:
        config_path.write_text(new_text, encoding="utf-8")
    return removed


def patch_repo(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--repo-id needs HF_TOKEN in the environment (Kaggle Secrets)")
    api = HfApi(token=token)
    local = hf_hub_download(repo_id, "config.json", repo_type="model", token=token)
    new_text, removed = _patch_text(Path(local).read_text(encoding="utf-8"))
    if not removed:
        return []
    tmp = Path(local).with_suffix(".patched.json")
    tmp.write_text(new_text, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="config.json",
                    repo_id=repo_id, repo_type="model",
                    commit_message="strip vLLM-incompatible scale_dtype/zp_dtype (llm-compressor#2057)")
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo-id", help="HF model repo to patch in place (download config.json, strip, re-upload)")
    g.add_argument("--local", type=Path, help="local model dir whose config.json to strip")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.local:
        target = args.local / "config.json"
        if not target.is_file():
            raise SystemExit(f"{target} not found")
        removed = patch_local(target)
        where = str(target)
    else:
        removed = patch_repo(args.repo_id)
        where = args.repo_id

    if removed:
        print(f"patched {where}: removed {len(removed)} key(s): {removed}")
    else:
        print(f"{where}: already clean (no scale_dtype/zp_dtype present) — no change")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
