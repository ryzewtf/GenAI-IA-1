"""Make a compressed-tensors ``config.json`` loadable by the PINNED consumer (vLLM 0.10.2).

Root cause (one bug class, several symptoms)
--------------------------------------------
The DeepSeek checkpoint is WRITTEN by ``llmcompressor==0.11.0`` → ``compressed-tensors==0.16.0``
(producer) but READ by vLLM 0.10.2, which pins ``compressed-tensors==0.11.0`` (consumer; verified in
vLLM's ``requirements/common.txt`` for the v0.10.2 tag). ct 0.11.0's ``QuantizationArgs`` is a pydantic
model with ``extra="forbid"`` and a fixed field allowlist, plus a ``@model_validator`` constraint — so
anything ct 0.16 added, or any value combination 0.11 disallows, is a HARD load failure. We cannot bump
vLLM: its version is baked into the vLLM panel's ``run_config_sha256`` / ``engine_build`` and a different
build would orphan the other six models' traces.

We have hit this wall TWICE on paid GPU, each a new symptom of the same skew:

  1. ``scale_dtype`` / ``zp_dtype`` (ct 0.16 additions, both ``None``) — not in ct 0.11's allowlist →
     ``Extra inputs are not permitted [extra_forbidden]`` (llm-compressor#2057).
  2. ``actorder: "static"`` on a per-CHANNEL (W8A8) weights group — ct 0.11's ``validate_model_after``
     raises *"Must use group quantization strategy in order to apply activation ordering"* because
     ``actorder`` is only legal with ``strategy == "group"``.

Both are inert for our build: the null dtypes carry no quant info, and ``actorder`` is a GPTQ
*processing-order* hint — for channel/tensor strategy no column permutation is stored (ct has no
``g_idx`` field, and channel-strategy GPTQ writes no ``weight_g_idx`` tensor), so nulling it changes
nothing the loader needs. The real quant data is ``config_groups`` (num_bits/type/symmetric/strategy) +
the ``weight_scale`` / ``weight_packed`` tensors. So this is a lossless CONFIG edit — NEVER a re-quant.

This script flushes the whole class at once, deterministically, on CPU:
  * recursively drop ``scale_dtype`` / ``zp_dtype`` anywhere in ``quantization_config``;
  * null ``actorder`` on any weights/activation args whose ``strategy`` is not ``group``;
  * drop any inner-args key NOT in ct 0.11.0's allowlist — but HARD-REFUSE if such a key is non-null
    and not one of the known-inert ones, so we never silently discard real quant data;
  * HARD-REFUSE if an ``actorder`` strip coincides with ``weight_g_idx`` tensors in the checkpoint
    (that would mean a real permutation we'd be orphaning — impossible for channel strategy, but guarded);
  * OPTIONALLY (when ``compressed_tensors`` is importable, i.e. a GPU/CPU env with torch) validate each
    patched group against the consumer's own ``QuantizationArgs`` and FAIL on any remaining rejection —
    so a brand-new skew symptom surfaces here (free CPU) instead of on a paid GPU session.

Two modes:
  * ``--repo-id R``  : download ``config.json`` (+ the safetensors index for the g_idx guard), strip,
                       re-upload just ``config.json`` (needs HF_TOKEN; the ~2h quant is NOT re-run).
                       Idempotent — re-running on an already-clean repo is a no-op.
  * ``--local DIR``  : strip ``DIR/config.json`` in place (reads ``DIR``'s index for the g_idx guard).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

#: ct 0.16 additions that ct 0.11's QuantizationArgs forbids; both are ``None`` here (llm-compressor#2057).
INCOMPATIBLE_KEYS = ("scale_dtype", "zp_dtype")

#: The EXACT field allowlist of compressed-tensors 0.11.0's ``QuantizationArgs`` (model_config
#: ``extra="forbid"``). Any inner-args key outside this set is a hard load failure on the consumer.
#: Source: compressed-tensors 0.11.0 ``src/compressed_tensors/quantization/quant_args.py``.
CT_0_11_ARG_FIELDS = frozenset({
    "num_bits", "type", "symmetric", "group_size", "strategy",
    "block_structure", "dynamic", "actorder", "observer", "observer_kwargs",
})

#: Keys we KNOW are inert (carry no quant info) and may therefore be dropped even when present. Any
#: OTHER non-allowlisted key triggers a hard refusal unless its value is null (nothing to lose).
KNOWN_INERT_EXTRA_KEYS = frozenset(INCOMPATIBLE_KEYS)

#: The three inner-args sub-dicts a config group may hold.
_ARG_SUBKEYS = ("weights", "input_activations", "output_activations")


def strip_incompatible_keys(config: dict[str, Any]) -> list[str]:
    """Remove INCOMPATIBLE_KEYS anywhere inside ``quantization_config`` (recursively). In place.

    Returns the dotted paths removed, so the caller can report exactly what changed (empty = already
    clean of these keys, a valid no-op).
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


def strip_illegal_actorder(config: dict[str, Any]) -> list[str]:
    """Null ``actorder`` on any config-group args whose ``strategy`` is not ``group``. In place.

    ct 0.11.0 raises *"Must use group quantization strategy in order to apply activation ordering"*
    unless ``strategy == "group"``. W8A8 int8 is per-CHANNEL, so ``actorder`` there is inert GPTQ
    metadata (no permutation is stored — see module docstring). Returns the dotted paths nulled.
    """
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return []
    removed: list[str] = []
    for gname, group in (qc.get("config_groups") or {}).items():
        if not isinstance(group, dict):
            continue
        for sub in _ARG_SUBKEYS:
            args = group.get(sub)
            if not isinstance(args, dict) or args.get("actorder") is None:
                continue
            strategy = str(args.get("strategy") or "").lower()
            if strategy != "group":
                args.pop("actorder")
                removed.append(f"quantization_config.config_groups.{gname}.{sub}.actorder")
    return removed


def flush_nonallowlisted_arg_keys(config: dict[str, Any]) -> list[str]:
    """Drop inner-args keys outside ct 0.11.0's allowlist. In place. Returns the dotted paths dropped.

    This is the general form of the two known symptoms: it catches ANY field ct 0.16 wrote that ct
    0.11 forbids (``extra="forbid"``), so a third unseen skew symptom never reaches a paid GPU. SAFETY:
    a dropped key with a non-null value that isn't in :data:`KNOWN_INERT_EXTRA_KEYS` is a HARD refusal —
    we strip inert cruft, never real quant data.
    """
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return []
    removed: list[str] = []
    for gname, group in (qc.get("config_groups") or {}).items():
        if not isinstance(group, dict):
            continue
        for sub in _ARG_SUBKEYS:
            args = group.get(sub)
            if not isinstance(args, dict):
                continue
            for key in list(args):
                if key in CT_0_11_ARG_FIELDS:
                    continue
                val = args[key]
                dotted = f"quantization_config.config_groups.{gname}.{sub}.{key}"
                if val is not None and key not in KNOWN_INERT_EXTRA_KEYS:
                    raise SystemExit(
                        f"REFUSING to patch: {dotted} = {val!r} is not a compressed-tensors 0.11.0 "
                        f"QuantizationArgs field and is non-null — dropping it could discard real quant "
                        f"data. Inspect the checkpoint; this is a NEW producer/consumer skew symptom.")
                args.pop(key)
                removed.append(dotted)
    return removed


def find_gidx_tensors(weight_keys: Sequence[str] | None) -> list[str]:
    """Tensor names that encode a GPTQ activation-ordering permutation (``*weight_g_idx`` / ``*.g_idx``).

    Channel/tensor-strategy GPTQ writes none; only group-strategy actorder does. Their presence means a
    real permutation exists in the weights, so nulling ``actorder`` in the config would orphan it.
    """
    if not weight_keys:
        return []
    return [k for k in weight_keys if k.endswith("g_idx") or k.endswith(".g_idx") or "weight_g_idx" in k]


def patch_config_dict(config: dict[str, Any], *, weight_keys: Sequence[str] | None = None) -> list[str]:
    """Apply all consumer-compat fixes to ``config`` in place. Returns every dotted path changed.

    Order: drop null dtype cruft → null illegal actorder → flush any other non-allowlisted inner-args
    key. If an actorder was nulled, refuse when ``weight_g_idx`` tensors exist (orphaned permutation).
    """
    removed = strip_incompatible_keys(config)
    actorder_removed = strip_illegal_actorder(config)
    removed += actorder_removed
    removed += flush_nonallowlisted_arg_keys(config)
    if actorder_removed:
        gidx = find_gidx_tensors(weight_keys)
        if gidx:
            raise SystemExit(
                "REFUSING to patch: nulled actorder but the checkpoint has activation-ordering "
                f"permutation tensors ({gidx[:3]}{'…' if len(gidx) > 3 else ''}). The weights are "
                "column-permuted; dropping actorder would misalign them. Re-quantize without actorder "
                "(or with group strategy) instead of patching.")
    return removed


#: The compressed-tensors version vLLM 0.10.2 pins (its requirements/common.txt). Only a match makes
#: validate_against_consumer's "OK" meaningful — the PRODUCER env (ct 0.16, present at quant save-time)
#: would "validate" its own output and prove nothing.
CONSUMER_CT_PIN = "0.11.0"


def validate_against_consumer(config: dict[str, Any]) -> str | None:
    """OPTIONAL post-patch check against the installed compressed-tensors schema. Raises on real failure.

    Only meaningful when the INSTALLED ct is the consumer pin (``0.11.0``, what vLLM 0.10.2 ships) — the
    best place to run this is the collect notebook, right after the vLLM install, in the shard-0 session.
    Builds each group's ``QuantizationArgs`` so any REMAINING rejection surfaces here (free CPU) not on a
    paid GPU. Returns a status note; raises SystemExit only if the pinned consumer still rejects the
    patched config. If ct is absent (torch-less dev box) or a DIFFERENT version (e.g. the producer's 0.16
    at quant save-time), it does NOT claim OK — it reports that the check was not against the pin.
    """
    try:
        from compressed_tensors.quantization.quant_args import QuantizationArgs  # noqa: PLC0415
        import compressed_tensors as ct  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 - torch-less dev box: skip, don't fail
        return f"consumer-validate SKIPPED (compressed_tensors not importable: {type(e).__name__})"

    qc = config.get("quantization_config") or {}
    n = 0
    for gname, group in (qc.get("config_groups") or {}).items():
        for sub in _ARG_SUBKEYS:
            args = (group or {}).get(sub)
            if not isinstance(args, dict):
                continue
            try:
                QuantizationArgs(**args)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"installed compressed_tensors {ct.__version__} STILL rejects "
                    f"config_groups.{gname}.{sub}: {e}") from e
            n += 1
    if ct.__version__ == CONSUMER_CT_PIN:
        return f"consumer-validate OK against compressed_tensors {ct.__version__} (the vLLM 0.10.2 pin; {n} group(s))"
    return (f"consumer-validate: {n} group(s) accepted by the INSTALLED compressed_tensors {ct.__version__}, "
            f"but that is NOT the vLLM 0.10.2 pin ({CONSUMER_CT_PIN}) — re-run this in the collect/vLLM env "
            "to truly confirm. (This run only proves the patched config parses under the installed ct.)")


def summarize_quant_args(config: dict[str, Any]) -> str:
    """One line per config-group arg dict, so the human can eyeball the REAL (never-before-seen) config.

    vLLM's int8-MoE method requires weights ``strategy: channel`` + inputs ``strategy: token, dynamic:
    true``; a two-line paste of this confirms the checkpoint matches that before a paid session.
    """
    qc = config.get("quantization_config") or {}
    lines: list[str] = []
    for gname, group in (qc.get("config_groups") or {}).items():
        for sub in _ARG_SUBKEYS:
            args = (group or {}).get(sub)
            if isinstance(args, dict):
                lines.append(f"  {gname}.{sub}: {json.dumps(args, sort_keys=True)}")
    return "cleaned quant args:\n" + ("\n".join(lines) if lines else "  (no config_groups)")


def _index_weight_keys(model_dir: Path) -> list[str] | None:
    """Tensor names from ``model.safetensors.index.json`` (for the g_idx guard). None if absent."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).keys())
    return None


def _patch_text(config_text: str, *,
                weight_keys: Sequence[str] | None = None) -> tuple[str, list[str], str | None, str]:
    config = json.loads(config_text)
    removed = patch_config_dict(config, weight_keys=weight_keys)
    note = validate_against_consumer(config)
    summary = summarize_quant_args(config)
    return json.dumps(config, indent=2) + "\n", removed, note, summary


def patch_local(config_path: Path, *, weight_keys: Sequence[str] | None = None) -> list[str]:
    """Strip ``config_path`` in place. ``weight_keys`` (or the dir's index) feeds the g_idx guard."""
    if weight_keys is None:
        weight_keys = _index_weight_keys(config_path.parent)
    new_text, removed, note, summary = _patch_text(
        config_path.read_text(encoding="utf-8"), weight_keys=weight_keys)
    print(summary, flush=True)
    if note:
        print(note, flush=True)
    if removed:
        config_path.write_text(new_text, encoding="utf-8")
    return removed


def patch_repo(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--repo-id needs HF_TOKEN in the environment (Kaggle Secrets)")
    api = HfApi(token=token)
    weight_keys: list[str] | None = None
    try:
        idx = hf_hub_download(repo_id, "model.safetensors.index.json", repo_type="model", token=token)
        weight_keys = list(json.loads(Path(idx).read_text(encoding="utf-8")).get("weight_map", {}).keys())
    except Exception as e:  # noqa: BLE001 - a single-file checkpoint has no index; the guard just sees None
        print(f"note: no safetensors index ({type(e).__name__}); g_idx guard will trust config only", flush=True)
    local = hf_hub_download(repo_id, "config.json", repo_type="model", token=token)
    new_text, removed, note, summary = _patch_text(
        Path(local).read_text(encoding="utf-8"), weight_keys=weight_keys)
    print(summary, flush=True)
    if note:
        print(note, flush=True)
    if not removed:
        return []
    tmp = Path(local).with_suffix(".patched.json")
    tmp.write_text(new_text, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="config.json",
                    repo_id=repo_id, repo_type="model",
                    commit_message="strip vLLM-0.10.2-incompatible quant config keys "
                                   "(scale_dtype/zp_dtype/illegal actorder; llm-compressor#2057)")
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
        print(f"{where}: already clean (no incompatible quant-config keys present) — no change")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
