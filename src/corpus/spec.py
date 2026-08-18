"""Corpus composition spec — plan T4.1.

Declares *what* the corpus is made of, separately from the code that assembles it. The split exists
because the composition is a claim the paper makes ("25/25/20/30 prose/code/math/multilingual") and
claims belong in a reviewable, hashable declaration rather than scattered through a build script.

Three rules here are not stylistic:

1. **FLORES-200 is a parallel control, never a volume source.** Its devtest split is ~1012 sentences
   per language. Asked to carry 30% of a 1M-token corpus it would have to be repeated dozens of
   times, and the multilingual condition would silently become "the same 1012 sentences, memorised".
   ``SourceSpec.role`` makes the distinction structural: a ``parallel_control`` source is capped.

2. **Substitutions are recorded, not silently applied.** The Stack is gated behind an access
   agreement, so code volume comes from ``starcoderdata``. That is a real difference in the code
   distribution and it goes in the artifact (T10.1), which means it has to survive in the spec.

3. **Token shares are targets under a reference counter, not guarantees per model.** The panel's
   tokenizers span 50k to 262k vocab, so one document is a different number of tokens for each
   checkpoint. The corpus file is shared by all seven (T4.3 requires identical splits), so the
   budget is set once under a reference counter and the *realized* per-model shares are measured
   and reported afterwards (``src/corpus/build.py:realized_shares``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

__all__ = [
    "DOMAINS",
    "TARGET_SHARES",
    "CorpusSpecError",
    "SourceSpec",
    "CorpusSpec",
    "DEFAULT_SOURCES",
    "MIXED_V1",
    "MIXED_V1_SCALE",
]

DOMAINS: tuple[str, ...] = ("prose", "code", "math", "multilingual")

TARGET_SHARES: dict[str, float] = {
    "prose": 0.25,
    "code": 0.25,
    "math": 0.20,
    "multilingual": 0.30,
}
"""Plan T4.1. Multilingual is the largest share because the routing literature's clearest
domain-specialisation claims are about language, so it carries the most weight in T9.4."""

Role = Literal["volume", "parallel_control"]

# A parallel control cannot exceed this share of its domain, no matter what the weights say. 1012
# sentences/lang is simply not enough text to be a volume source; see rule 1 in the module docstring.
PARALLEL_CONTROL_MAX_DOMAIN_SHARE = 0.20


class CorpusSpecError(Exception):
    """The composition is unbuildable or internally inconsistent. Always fatal, never a warning."""


@dataclass(frozen=True)
class SourceSpec:
    """One dataset feeding one domain."""

    name: str
    domain: str
    hf_dataset: str
    role: Role = "volume"
    hf_config: str | None = None
    hf_split: str = "train"
    langs: tuple[str, ...] = ()
    weight: float = 1.0
    """Relative weight *within* the domain. Normalised across the domain's sources."""
    license_note: str = ""
    substituted_for: str | None = None
    """Set when this source stands in for one the plan named but that is not obtainable."""
    note: str = ""

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise CorpusSpecError(f"source {self.name!r}: unknown domain {self.domain!r}")
        if self.weight <= 0:
            raise CorpusSpecError(f"source {self.name!r}: weight must be positive")
        if self.role == "parallel_control" and not self.langs:
            raise CorpusSpecError(
                f"source {self.name!r}: a parallel control exists to compare across languages, "
                "so it must declare which ones"
            )


# The Stack is gated; starcoderdata is the substitute and the substitution is part of the record.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="fineweb-edu",
        domain="prose",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        license_note="ODC-By-1.0",
        note="English prose volume. Sampled, never read end to end.",
    ),
    SourceSpec(
        name="starcoderdata",
        domain="code",
        hf_dataset="bigcode/starcoderdata",
        substituted_for="bigcode/the-stack",
        license_note="permissively licensed subset; see dataset card",
        note="The Stack requires an access agreement, which a headless Kaggle session cannot "
             "complete. The code distribution therefore differs from the plan's first choice.",
    ),
    SourceSpec(
        name="open-web-math",
        domain="math",
        hf_dataset="open-web-math/open-web-math",
        license_note="ODC-By-1.0",
        note="Mathematical prose and LaTeX, not just symbolic expressions.",
    ),
    SourceSpec(
        name="culturax",
        domain="multilingual",
        hf_dataset="uonlp/CulturaX",
        role="volume",
        langs=("de", "fr", "es", "ru", "zh", "ja", "ar", "hi"),
        weight=4.0,
        license_note="mC4 + OSCAR terms; per-language subsets",
        note="Volume source for the multilingual domain. mC4 is the documented alternative.",
    ),
    SourceSpec(
        name="flores200",
        domain="multilingual",
        hf_dataset="facebook/flores",
        role="parallel_control",
        hf_split="devtest",
        langs=("de", "fr", "es", "ru", "zh", "ja", "ar", "hi"),
        weight=1.0,
        license_note="CC-BY-SA-4.0",
        note="Parallel control: the SAME sentences in every language, which is what makes a "
             "language effect separable from a content effect. ~1012 sentences/lang, so capped.",
    ),
)


@dataclass(frozen=True)
class CorpusSpec:
    """A complete, buildable corpus definition."""

    name: str
    target_tokens: int
    max_doc_tokens: int = 2048
    """Plan T4.2 / I4. Equal to the pinned n_ctx: a longer document would be truncated at capture."""
    shard_tokens: int = 50_000
    """Plan T4.2. Shards never straddle a document, so this is a target, not an exact size."""
    split_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    seed: int = 0
    shares: Mapping[str, float] = field(default_factory=lambda: dict(TARGET_SHARES))
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES
    models: tuple[str, ...] = ()
    """Empty means "all seven". mixed-v1-scale is OLMoE-only (T5.4)."""

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise CorpusSpecError("target_tokens must be positive")
        if self.max_doc_tokens <= 0 or self.shard_tokens < self.max_doc_tokens:
            # A shard smaller than one document would force either straddling or one-doc shards.
            raise CorpusSpecError(
                f"shard_tokens ({self.shard_tokens}) must be at least max_doc_tokens "
                f"({self.max_doc_tokens}); shards must be able to hold a whole document"
            )

        total = sum(self.shares.values())
        if abs(total - 1.0) > 1e-9:
            raise CorpusSpecError(f"domain shares sum to {total!r}, not 1.0")
        unknown = set(self.shares) - set(DOMAINS)
        if unknown:
            raise CorpusSpecError(f"unknown domain(s) in shares: {sorted(unknown)}")
        missing = {d for d, s in self.shares.items() if s > 0} - {s.domain for s in self.sources}
        if missing:
            raise CorpusSpecError(f"domain(s) with a nonzero share but no source: {sorted(missing)}")

        ratio_total = sum(self.split_ratios.values())
        if abs(ratio_total - 1.0) > 1e-9:
            raise CorpusSpecError(f"split ratios sum to {ratio_total!r}, not 1.0")
        if set(self.split_ratios) != {"train", "val", "test"}:
            raise CorpusSpecError(f"splits must be exactly train/val/test, got {sorted(self.split_ratios)}")

        # Rule 1: a parallel control must not become the multilingual condition by volume.
        for domain in DOMAINS:
            in_domain = [s for s in self.sources if s.domain == domain]
            if not in_domain:
                continue
            denom = sum(s.weight for s in in_domain)
            for source in in_domain:
                if source.role != "parallel_control":
                    continue
                share = source.weight / denom
                if share > PARALLEL_CONTROL_MAX_DOMAIN_SHARE + 1e-9:
                    raise CorpusSpecError(
                        f"{source.name!r} is a parallel_control but is weighted for "
                        f"{share:.0%} of the {domain!r} domain (cap "
                        f"{PARALLEL_CONTROL_MAX_DOMAIN_SHARE:.0%}). FLORES-200 devtest is ~1012 "
                        "sentences per language; at this weight the multilingual condition would "
                        "be the same handful of sentences repeated. Add a volume source instead."
                    )

    # -- derived quantities -------------------------------------------------------------------

    def domain_token_targets(self) -> dict[str, int]:
        """Reference-counter token budget per domain. Largest domain absorbs the rounding."""
        raw = {d: int(round(self.target_tokens * s)) for d, s in self.shares.items() if s > 0}
        drift = self.target_tokens - sum(raw.values())
        if drift and raw:
            biggest = max(raw, key=lambda d: raw[d])
            raw[biggest] += drift
        return raw

    def source_token_targets(self) -> dict[str, int]:
        """Token budget per source: the domain budget split by within-domain weight."""
        out: dict[str, int] = {}
        for domain, budget in self.domain_token_targets().items():
            in_domain = [s for s in self.sources if s.domain == domain]
            denom = sum(s.weight for s in in_domain)
            allocated = 0
            for i, source in enumerate(in_domain):
                if i == len(in_domain) - 1:
                    out[source.name] = budget - allocated  # exact, no rounding leak
                else:
                    take = int(round(budget * source.weight / denom))
                    out[source.name] = take
                    allocated += take
        return out

    def substitutions(self) -> list[dict[str, str]]:
        """What we could not obtain and what stands in for it (T10.1 must publish this)."""
        return [
            {"source": s.name, "substituted_for": s.substituted_for or "", "note": s.note}
            for s in self.sources
            if s.substituted_for
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(s) for s in self.sources]
        payload["domain_token_targets"] = self.domain_token_targets()
        payload["source_token_targets"] = self.source_token_targets()
        payload["substitutions"] = self.substitutions()
        return payload

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


MIXED_V1 = CorpusSpec(name="mixed-v1", target_tokens=1_000_000)
"""The main corpus, all seven checkpoints. 500k if gate Q1 fires (T0.5)."""

MIXED_V1_SCALE = CorpusSpec(
    name="mixed-v1-scale",
    target_tokens=4_000_000,
    models=("olmoe-0125",),
    seed=1,
)
"""T5.4 sample-size sensitivity, OLMoE only. A different seed so it is not a superset of mixed-v1 --
if it were, "more data" and "the same data again" would be indistinguishable in the result."""
