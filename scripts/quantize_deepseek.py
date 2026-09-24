"""Self-quantize DeepSeek-V2-Lite to W4A16 (GPTQ) with the router kept in fp16.

Why this exists (plan D2, Tier 2)
---------------------------------
DeepSeek-V2-Lite is 15.7B params — fp16 is ~31 GB and does not fit one T4, and no Turing-viable
public INT4 build exists. So we self-quantize the routed experts (and attention/dense FFN) to 4-bit
weights with 16-bit activations (W4A16) using GPTQModel, which fits one T4 at ~8 GB, and we KEEP THE
ROUTER (``.mlp.gate``) in fp16. The router is the study's measured object — a 4-bit router would
degrade expert selection, and vLLM's ``DeepseekV2ForCausalLM.load_weights`` expects an unquantized
router anyway. This mirrors qwen3-30b (already a Tier-1 model on a public GPTQ-Int4 build whose
router is fp16) so quant format stays a documented per-model difference, not a router change.

The skip is done with GPTQModel's ``QuantizeConfig.dynamic`` negative-match: ``-:<regex>`` excludes
matching modules from quantization. ``ROUTER_SKIP_PATTERN`` matches every layer's ``.mlp.gate`` and
NOTHING else — not ``.mlp.experts`` (routed), ``.mlp.shared_experts``, or the dense layer-0
``.mlp.gate_up_proj``. Those regex facts are unit-tested without importing GPTQModel.

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

# Negative-match: skip the MoE router in every layer. `-:` is GPTQModel's "exclude" prefix; the regex
# is matched against module names like `model.layers.7.mlp.gate`. `$` anchors the end so it matches
# ONLY the router gate, never `.mlp.gate_up_proj` (dense/shared FFN) which starts the same way.
ROUTER_SKIP_PATTERN = r"-:.*\.mlp\.gate$"


def router_gate_is_skipped(module_name: str) -> bool:
    """True iff ``module_name`` is a router gate excluded by :data:`ROUTER_SKIP_PATTERN`.

    Pure and dependency-free so the skip rule is unit-testable without GPTQModel. Mirrors GPTQModel's
    negative-match: strip the leading ``-:`` and full-match the remaining regex against the name.
    """
    body = ROUTER_SKIP_PATTERN[2:] if ROUTER_SKIP_PATTERN.startswith("-:") else ROUTER_SKIP_PATTERN
    return re.fullmatch(body, module_name) is not None


def dynamic_skip_config() -> dict[str, dict]:
    """The ``QuantizeConfig.dynamic`` mapping that keeps the router fp16 (skip = empty override)."""
    return {ROUTER_SKIP_PATTERN: {}}


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

    Scans the safetensors weight index: a quantized module carries ``.qweight``; the router gate must
    NOT, and the routed experts MUST. This is the last gate before the weights are trusted/uploaded.
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

    gate_q = [k for k in keys if re.search(r"\.mlp\.gate\.qweight$", k)]
    experts_q = [k for k in keys if ".mlp.experts" in k and k.endswith(".qweight")]
    if gate_q:
        raise SystemExit(f"router gate was QUANTIZED (found {len(gate_q)} .mlp.gate.qweight) — the "
                         "dynamic skip did not apply; refusing to upload a W4 router")
    if not experts_q:
        raise SystemExit("no quantized experts found (.mlp.experts.*.qweight) — quantization did not "
                         "run over the routed experts; refusing to upload")
    print(f"verify OK: 0 quantized router gates, {len(experts_q)} quantized expert tensors")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Self-quantize DeepSeek-V2-Lite to W4A16 (router fp16)")
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--out", type=Path, required=True, help="local output dir for the quantized model")
    p.add_argument("--repo-id", default="Ryze242005/DeepSeek-V2-Lite-w4a16-gptq")
    p.add_argument("--calib-corpus", type=Path, default=None,
                   help="JSONL with a 'text' field (the mounted mixed-v2 corpus); wikitext if omitted")
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--push", action="store_true", help="push to --repo-id (private) after verify")
    p.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = p.parse_args(list(argv) if argv is not None else None)

    print(f"# quantize {args.model_id} -> W4A16 (bits={args.bits}, group_size={args.group_size})")
    print(f"# router kept fp16 via dynamic skip: {ROUTER_SKIP_PATTERN}")
    print(f"# out={args.out}  repo={args.repo_id}  push={args.push}")
    if args.dry_run:
        return 0

    texts = load_calibration_texts(args.calib_corpus, args.calib_samples)

    from gptqmodel import GPTQModel, QuantizeConfig  # noqa: PLC0415 - GPU env only

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=True,
        sym=True,
        dynamic=dynamic_skip_config(),  # keep the router (.mlp.gate) in fp16
    )
    model = GPTQModel.load(args.model_id, qcfg, trust_remote_code=True)
    # GPTQModel accepts raw strings and tokenizes with the model's tokenizer.
    model.quantize([t[: args.seq_len * 6] for t in texts], batch_size=1)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"saved quantized model to {args.out}")

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
