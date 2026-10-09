"""Self-quantize DeepSeek-V2-Lite to W4A16 (GPTQ via llm-compressor) with the router kept fp16.

Why this exists (plan D2, Tier 2)
---------------------------------
DeepSeek-V2-Lite is 15.7B params — fp16 is ~31 GB and does not fit one T4, and no Turing-viable
public INT4 build exists. So we self-quantize the routed experts (and attention/dense FFN) to 4-bit
weights with 16-bit activations (W4A16) and KEEP THE ROUTER (``.mlp.gate``) in fp16. The router is
the study's measured object — a 4-bit router would degrade expert selection, and vLLM expects an
unquantized router anyway. This mirrors qwen3-30b (a Tier-1 model on a public GPTQ-Int4 build whose
router is fp16) so quant format stays a documented per-model difference, not a router change.

Engine = llm-compressor (vLLM's own PTQ library), NOT GPTQModel
---------------------------------------------------------------
GPTQModel is sdist-only and its CUDA kernels FAIL to compile on Kaggle's Python 3.13 image (burned
two sessions). llm-compressor ships a pure-Python wheel (no compile) and — in the 0.13.x line — pins
``transformers>=4.56.1,<=4.57.6``. That transformers range is load-bearing: transformers 5.0 removed
``is_torch_fx_available`` (which DeepSeek-V2's trust_remote_code modeling imports) AND silently
mis-tokenizes DeepSeek, which would corrupt calibration. 4.57.6 keeps both correct. Install
``llmcompressor==0.13.0`` (see notebooks/quantize_deepseek.ipynb). Output is compressed-tensors,
which vLLM loads natively (no --quantization flag).

The router stays fp16 via the GPTQModifier ``ignore`` list: ``IGNORE_PATTERNS`` matches every
layer's ``.mlp.gate`` and the lm_head, and NOTHING else — not ``.mlp.experts`` (routed),
``.mlp.shared_experts``, or any ``.mlp.gate_proj``/``.mlp.gate_up_proj`` (dense/shared FFN). Those
regex facts are unit-tested without importing llm-compressor.

Output repo: ``Ryze242005/DeepSeek-V2-Lite-w4a16-gptq`` (private). model_sha256 stays null in
models.yaml until this repo exists; the collector then resolves + pins the HF revision sha.

Run on Kaggle T4x2 via ``notebooks/quantize_deepseek.ipynb``. HF write token from Kaggle Secrets as
``HF_TOKEN`` (env) — NEVER inline it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

# llm-compressor's ``ignore`` is a list of module-name matchers; a ``re:`` prefix makes the rest a
# regex matched against names like ``model.layers.7.mlp.gate``. ``$`` anchors the end so the router
# pattern matches ONLY the MoE gate, never ``.mlp.gate_proj``/``.mlp.gate_up_proj`` (which start the
# same way). ``lm_head`` is kept fp16 too (standard — a 4-bit output head produces gibberish).
ROUTER_IGNORE_REGEX = r"re:.*\.mlp\.gate$"
# DeepSeek-V2-Lite's DENSE layer 0 (first_k_dense_replace=1) FFN has intermediate_size=10944, whose
# down_proj has 10944 input columns — NOT divisible by the group size 128 (10944/128 = 85.5), so
# group-128 W4A16 cannot tile it and llm-compressor refuses. The MoE experts use moe_intermediate
# 1408 = 11*128, which divides fine, so only this one dense layer is affected. Layer 0 is dense (not a
# MoE layer) and is excluded from the trace anyway (moe_layer_offset=1 → trace layer 0 == model layer
# 1), so keeping its FFN fp16 is inconsequential to the study. Match its gate/up/down projections.
DENSE_LAYER0_IGNORE_REGEX = r"re:model\.layers\.0\.mlp\.\w+_proj$"
IGNORE_PATTERNS = ["lm_head", ROUTER_IGNORE_REGEX, DENSE_LAYER0_IGNORE_REGEX]


def router_gate_is_ignored(module_name: str) -> bool:
    """True iff ``module_name`` is a router gate kept fp16 by :data:`ROUTER_IGNORE_REGEX`.

    Pure and dependency-free so the skip rule is unit-testable without llm-compressor. Mirrors
    llm-compressor's matcher: strip the leading ``re:`` and full-match the regex against the name.
    """
    body = ROUTER_IGNORE_REGEX[3:] if ROUTER_IGNORE_REGEX.startswith("re:") else ROUTER_IGNORE_REGEX
    return re.fullmatch(body, module_name) is not None


def module_is_ignored(module_name: str) -> bool:
    """True iff ``module_name`` is kept fp16 by ANY entry in :data:`IGNORE_PATTERNS`.

    Mirrors llm-compressor's matching for the whole ignore list: an ``re:``-prefixed entry is a regex
    (full-match), a bare entry matches the module's final ``.``-segment (so ``lm_head`` matches
    ``model.lm_head``). Pure/dependency-free so the full fp16 set is unit-testable without a GPU.
    """
    leaf = module_name.rsplit(".", 1)[-1]
    for pat in IGNORE_PATTERNS:
        if pat.startswith("re:"):
            if re.fullmatch(pat[3:], module_name) is not None:
                return True
        elif pat == module_name or pat == leaf:
            return True
    return False


def load_calibration_texts(corpus: Path | None, n_samples: int) -> list[str]:
    """Calibration strings: a sample of the study corpus if given, else a wikitext fallback.

    Using the actual corpus keeps the calibration distribution matched to what we collect against.
    Reads at most ``n_samples`` non-empty ``text`` fields; falls back to HF ``wikitext`` only if no
    corpus path is provided (the offline Kaggle run should always pass the mounted corpus).
    """
    if corpus is not None:
        texts: list[str] = []
        with Path(corpus).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                t = (json.loads(line).get("text") or "").strip()
                if t:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
        if not texts:
            raise SystemExit(f"no usable 'text' rows in {corpus}")
        print(f"calibration: {len(texts)} texts from {corpus}")
        return texts

    from datasets import load_dataset  # noqa: PLC0415 - only the fallback needs it

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t and t.strip()][:n_samples]
    print(f"calibration: {len(texts)} texts from wikitext-2-raw-v1 (fallback)")
    return texts


def _weight_keys(out_dir: Path) -> list[str]:
    """Every tensor name in the saved checkpoint (sharded index or single file)."""
    index = out_dir / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).keys())
    from safetensors import safe_open  # noqa: PLC0415

    single = out_dir / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"no safetensors index or file under {out_dir}")
    with safe_open(str(single), framework="numpy") as f:
        return list(f.keys())


def assess_quant(keys: list[str], *, ignore: list, config_groups: dict | None) -> tuple[bool, str]:
    """Pure verdict on a saved compressed-tensors checkpoint. Returns (ok, message).

    Scheme-agnostic: a quantized Linear is marked by ``.weight_packed`` (W4A16 pack-quantized, sub-byte)
    OR ``.weight_scale`` (W8A8 int8, stored as int8 ``.weight`` + a scale) — we key on BOTH so a W8A8
    checkpoint (int8, Turing-viable) verifies as well as a W4A16 one.

    HARD FAIL only on genuine defects, so a surprising-but-fine checkpoint is never rejected (that
    would waste a multi-hour run):
      * router QUANTIZED — any ``.mlp.gate.weight_packed`` OR ``.mlp.gate.weight_scale`` tensor. This
        is the one thing we must never ship, so its mere presence fails.
      * nothing compressed — no ``config_groups`` in quantization_config AND no ``.weight_packed`` /
        ``.weight_scale`` tensor anywhere: the oneshot did not quantize, so there is nothing to upload.
    Everything else (expert count, exact key spelling) is reported but does not fail.
    """
    gate_q = [k for k in keys
              if re.search(r"\.mlp\.gate\.weight_packed$", k) or re.search(r"\.mlp\.gate\.weight_scale$", k)]
    experts_q = [k for k in keys
                 if ".mlp.experts" in k and (k.endswith(".weight_packed") or k.endswith(".weight_scale"))]
    any_q = [k for k in keys if k.endswith(".weight_packed") or k.endswith(".weight_scale")]
    router_in_ignore = any(
        "mlp.gate" in str(e) and "gate_proj" not in str(e) and "gate_up" not in str(e)
        for e in (ignore or [])
    )
    if gate_q:
        return False, (f"router gate was QUANTIZED ({len(gate_q)} .mlp.gate.weight_packed/_scale) — the "
                       "ignore pattern did not apply; refusing to upload a quantized router")
    if not config_groups and not any_q:
        return False, ("nothing was quantized (no config_groups in quantization_config and no "
                       ".weight_packed/.weight_scale tensors) — the oneshot did not run; refusing to upload")
    return True, (f"router fp16 (0 quantized .mlp.gate, router_in_ignore={router_in_ignore}); "
                  f"{len(experts_q)} quantized expert tensors, {len(any_q)} quantized total")


def verify_router_unquantized(out_dir: Path) -> None:
    """Last gate before upload: router stayed fp16 and the model was actually compressed.

    Reads config.json's ``quantization_config`` (authoritative for what stayed fp16 — the ``ignore``
    list and ``config_groups``) plus the saved tensor names, and defers to :func:`assess_quant`.
    """
    cfg_path = out_dir / "config.json"
    qc: dict = {}
    if cfg_path.is_file():
        qc = (json.loads(cfg_path.read_text(encoding="utf-8")).get("quantization_config") or {})
    ok, msg = assess_quant(_weight_keys(out_dir),
                           ignore=qc.get("ignore") or [], config_groups=qc.get("config_groups"))
    if not ok:
        raise SystemExit(msg)
    print(f"verify OK: {msg}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Self-quantize DeepSeek-V2-Lite to W4A16 (router fp16)")
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--out", type=Path, required=True, help="local output dir for the quantized model")
    p.add_argument("--repo-id", default="Ryze242005/DeepSeek-V2-Lite-w8a8-int8")
    # SCHEME MUST be a Turing-viable one. vLLM runs compressed-tensors WNA16 (W4A16 / W8A16) ONLY via
    # the Marlin kernel, which requires compute capability 80 (Ampere) — it HARD-FAILS on the T4 (sm_75)
    # the campaign runs on ("Min capability: 80. Current capability: 75", on a dense attn Linear, during
    # load). INT8 W8A8 uses the cutlass int8 GEMM (dense, min_capability=75) + the Triton fused_experts
    # int8 MoE path (no capability guard) — both run on Turing. So default is W8A8, NOT W4A16/W8A16.
    # Router + lm_head stay fp16 via ignore regardless of scheme (the measured object is untouched).
    p.add_argument("--scheme", default="W8A8", choices=["W8A8", "W4A16"],
                   help="W8A8 (int8 w+a, Turing-viable, DEFAULT) or W4A16 (int4, Marlin/sm_80-only — "
                        "DEAD on T4, kept only for Ampere+ boxes).")
    p.add_argument("--calib-corpus", type=Path, default=None,
                   help="JSONL with a 'text' field (the mounted mixed-v2 corpus); wikitext if omitted")
    p.add_argument("--calib-samples", type=int, default=128,
                   help="calibration docs. 128 is a standard GPTQ size; fewer = smaller CPU-side "
                        "activation cache and faster. Router stays fp16, so this only affects expert "
                        "fidelity, not the measured routing.")
    p.add_argument("--group-size", type=int, default=128,
                   help="only used by W4A16 (group-128). W8A8 int8 is per-channel; this is ignored.")
    p.add_argument("--seq-len", type=int, default=1024,
                   help="calibration sequence length. Activation-cache RAM scales with samples*seq; "
                        "1024 halves it vs 2048 with negligible W4A16 quality cost.")
    # DeepSeek-V2-Lite is ~15.7B (~31GB bf16) — it does NOT fit in Kaggle's ~30GB host RAM alongside
    # quantization overhead, which OOM-killed (exit -9) the fully-resident run at layer 18/26. These
    # spill the model across BOTH T4s + a bounded CPU slice + disk, so host RAM can never fill. The GPU
    # caps are resident-weight budgets only (leaving headroom for the sequential pipeline's per-layer
    # onload + Hessian scratch); runtime allocations may exceed them up to the 15GB physical limit.
    p.add_argument("--offload-dir", type=Path, default=Path("/kaggle/working/offload"),
                   help="disk folder for weights that don't fit in the GPU+CPU budget (Kaggle disk).")
    p.add_argument("--max-gpu-mem", default="11GiB",
                   help="per-GPU resident-weight cap (T4=15GB usable; the rest is onload/Hessian room).")
    p.add_argument("--max-cpu-mem", default="8GiB",
                   help="host-RAM cap for resident weights — kept well under Kaggle's ~29GB so the "
                        "CPU-side activation cache + Python can't OOM.")
    p.add_argument("--no-offload", action="store_true",
                   help="load fully resident (torch_dtype=auto, no device_map) — only for a box whose "
                        "RAM comfortably exceeds the model; OOMs on Kaggle.")
    p.add_argument("--push", action="store_true", help="push to --repo-id (private) after verify")
    p.add_argument("--all-experts", action="store_true",
                   help="route every calibration token through ALL experts (moe_calibrate_all_experts). "
                        "OOMs Kaggle's ~30GB host RAM on V2-Lite (64 experts x 26 layers) — default OFF.")
    p.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = p.parse_args(list(argv) if argv is not None else None)

    print(f"# quantize {args.model_id} -> {args.scheme} GPTQ via llm-compressor"
          f"{f' (group_size={args.group_size})' if args.scheme == 'W4A16' else ' (int8 per-channel)'}")
    print(f"# router + lm_head kept fp16 via ignore: {IGNORE_PATTERNS}")
    print(f"# out={args.out}  repo={args.repo_id}  push={args.push}  all_experts={args.all_experts}")
    print(f"# calib: {args.calib_samples} samples x {args.seq_len} tok  "
          f"| load={'resident' if args.no_offload else 'offload(2xGPU+CPU+disk)'}")
    if args.dry_run:
        return 0

    texts = load_calibration_texts(args.calib_corpus, args.calib_samples)

    # GPU-env-only imports (kept out of module import so the unit tests stay torch-free).
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import GPTQModifier  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if args.no_offload:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype="auto", trust_remote_code=True)
    else:
        # Spill the ~31GB model across both T4s + a bounded CPU slice + disk. accelerate's device_map
        # places layers to fit this budget; the llm-compressor sequential pipeline then onloads the
        # active layer to the GPU for compute (align_module_device is accelerate-hook-aware), so a
        # dispatched/offloaded model is supported. CPU is capped low so host RAM cannot OOM.
        import torch  # noqa: PLC0415

        args.offload_dir.mkdir(parents=True, exist_ok=True)
        n_gpu = torch.cuda.device_count()
        max_memory = {i: args.max_gpu_mem for i in range(n_gpu)}
        max_memory["cpu"] = args.max_cpu_mem
        print(f"# offload load: device_map=auto  max_memory={max_memory}  disk={args.offload_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype="auto", trust_remote_code=True,
            device_map="auto", max_memory=max_memory,
            offload_folder=str(args.offload_dir), offload_state_dict=True)
        dev_map = getattr(model, "hf_device_map", None)
        if dev_map:
            where = {str(v) for v in dev_map.values()}
            print(f"# dispatched across: {sorted(where)} ({len(dev_map)} modules)")

    ds = Dataset.from_dict({"text": texts})
    # Scheme presets fix their own layout, and GPTQModifier rejects a separate group_size kwarg
    # (pydantic extra_forbidden). W4A16 = int4 group-128 (hence the 128 guard + the dense-layer-0
    # ignore, since 10944 isn't divisible by 128). W8A8 = int8 per-channel weights + dynamic int8
    # activations — no group size, so the divisibility issue does not arise. Router + lm_head stay fp16
    # via IGNORE_PATTERNS either way.
    if args.scheme == "W4A16" and args.group_size != 128:
        raise SystemExit(f"--group-size {args.group_size}: the W4A16 scheme is fixed at group_size=128; "
                         "pass config_groups in the recipe to change it.")
    recipe = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=IGNORE_PATTERNS)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        tokenizer=tokenizer,
        max_seq_length=args.seq_len,
        num_calibration_samples=len(texts),
        trust_remote_code_model=True,
        # MoE calibration coverage. all-experts=True routes EVERY token through ALL 64 experts/layer,
        # which caches 64x the activations and OOMs Kaggle's ~30GB host RAM partway through (died at
        # layer 18/26 after ~2h). Default False = standard GPTQ: each expert is calibrated only on the
        # tokens the router actually sends it — far less memory, ~10x faster. The router (the measured
        # object) is fp16 regardless, so expert-calibration coverage does not change which experts get
        # selected; it only affects expert-output fidelity, which is acceptable for a routing study.
        moe_calibrate_all_experts=args.all_experts,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), save_compressed=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"saved compressed-tensors model to {args.out}")

    # compressed-tensors 0.16 writes scale_dtype/zp_dtype (null) into config.json, which the PINNED
    # vLLM 0.10.2 rejects at engine-config parse (llm-compressor#2057). Strip them before push so the
    # checkpoint loads on the campaign engine; lossless (both are null). See scripts/patch_quant_config.
    from scripts.patch_quant_config import patch_local  # noqa: PLC0415

    stripped = patch_local(args.out / "config.json")
    if stripped:
        print(f"stripped vLLM-incompatible quant keys from config.json: {stripped}")

    verify_router_unquantized(args.out)

    if args.push:
        import os  # noqa: PLC0415

        from huggingface_hub import HfApi  # noqa: PLC0415

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--push needs HF_TOKEN in the environment (Kaggle Secrets)")
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.repo_id, repo_type="model")
        print(f"pushed to https://huggingface.co/{args.repo_id} (private)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
