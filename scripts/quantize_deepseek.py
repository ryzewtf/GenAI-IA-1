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
IGNORE_PATTERNS = ["lm_head", ROUTER_IGNORE_REGEX]


def router_gate_is_ignored(module_name: str) -> bool:
    """True iff ``module_name`` is a router gate kept fp16 by :data:`ROUTER_IGNORE_REGEX`.

    Pure and dependency-free so the skip rule is unit-testable without llm-compressor. Mirrors
    llm-compressor's matcher: strip the leading ``re:`` and full-match the regex against the name.
    """
    body = ROUTER_IGNORE_REGEX[3:] if ROUTER_IGNORE_REGEX.startswith("re:") else ROUTER_IGNORE_REGEX
    return re.fullmatch(body, module_name) is not None


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


def verify_router_unquantized(out_dir: Path) -> None:
    """Fail loudly if the saved model quantized any router gate, or quantized no experts at all.

    compressed-tensors stores a quantized Linear's 4-bit payload as ``<module>.weight_packed`` (plus
    ``.weight_scale``); an unquantized module keeps a plain ``.weight``. So the router gate must NOT
    carry ``.weight_packed`` and the routed experts MUST. This is the last gate before upload.
    """
    index = out_dir / "model.safetensors.index.json"
    keys: list[str]
    if index.is_file():
        keys = list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).keys())
    else:
        # single-file checkpoint: read tensor names from the safetensors header
        from safetensors import safe_open  # noqa: PLC0415

        single = out_dir / "model.safetensors"
        if not single.is_file():
            raise SystemExit(f"no safetensors index or file under {out_dir}")
        with safe_open(str(single), framework="numpy") as f:
            keys = list(f.keys())

    gate_q = [k for k in keys if re.search(r"\.mlp\.gate\.weight_packed$", k)]
    experts_q = [k for k in keys if ".mlp.experts" in k and k.endswith(".weight_packed")]
    if gate_q:
        raise SystemExit(f"router gate was QUANTIZED (found {len(gate_q)} .mlp.gate.weight_packed) — "
                         "the ignore pattern did not apply; refusing to upload a W4 router")
    if not experts_q:
        raise SystemExit("no quantized experts found (.mlp.experts.*.weight_packed) — quantization "
                         "did not run over the routed experts; refusing to upload")
    print(f"verify OK: 0 quantized router gates, {len(experts_q)} quantized expert tensors")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Self-quantize DeepSeek-V2-Lite to W4A16 (router fp16)")
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--out", type=Path, required=True, help="local output dir for the quantized model")
    p.add_argument("--repo-id", default="Ryze242005/DeepSeek-V2-Lite-w4a16-gptq")
    p.add_argument("--calib-corpus", type=Path, default=None,
                   help="JSONL with a 'text' field (the mounted mixed-v2 corpus); wikitext if omitted")
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--push", action="store_true", help="push to --repo-id (private) after verify")
    p.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = p.parse_args(list(argv) if argv is not None else None)

    print(f"# quantize {args.model_id} -> W4A16 GPTQ via llm-compressor (group_size={args.group_size})")
    print(f"# router + lm_head kept fp16 via ignore: {IGNORE_PATTERNS}")
    print(f"# out={args.out}  repo={args.repo_id}  push={args.push}")
    if args.dry_run:
        return 0

    texts = load_calibration_texts(args.calib_corpus, args.calib_samples)

    # GPU-env-only imports (kept out of module import so the unit tests stay torch-free).
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import GPTQModifier  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", trust_remote_code=True)

    ds = Dataset.from_dict({"text": texts})
    recipe = GPTQModifier(
        targets="Linear", scheme="W4A16", ignore=IGNORE_PATTERNS, group_size=args.group_size)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        tokenizer=tokenizer,
        max_seq_length=args.seq_len,
        num_calibration_samples=len(texts),
        trust_remote_code_model=True,
        # MoE: route calibration tokens through EVERY expert, not just the ones the router picks, so
        # no expert is left uncalibrated (the default, pinned here so a version change can't flip it).
        moe_calibrate_all_experts=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), save_compressed=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"saved compressed-tensors model to {args.out}")

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
