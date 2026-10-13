"""CPU tests for the Phase 9/10 drivers' pure logic (no traces needed)."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tokenizers")  # phase9_word_f1 imports it at module level

from scripts import phase9_domain, phase9_sample_size, phase9_word_f1, phase8_conclusions  # noqa: E402
from src.probes.features import CorpusIndex  # noqa: E402


def _index(token_ids, doc_ids, pos, train, val, test):
    n = len(token_ids)
    return CorpusIndex(
        token_ids=np.asarray(token_ids), doc_ids=np.asarray(doc_ids), pos_in_doc=np.asarray(pos),
        split_rows={"train": np.asarray(train), "val": np.asarray(val), "test": np.asarray(test)},
        captured=np.arange(n), vocab_size=max(token_ids) + 1,
    )


def test_word_ids_marks_word_initial_and_joins_pieces():
    pieces = ["Hello", " wor", "ld", " again", "!"]
    idx = _index([0, 1, 2, 3, 4], [0] * 5, [0, 1, 2, 3, 4], [0], [1], [2])
    initial, wid = phase9_word_f1.word_ids(idx, pieces)
    assert initial.tolist() == [True, True, False, True, False]
    assert wid[2] == -1 and wid[4] == -1
    assert len({wid[0], wid[1], wid[3]}) == 3


def test_word_ids_same_word_same_id_and_doc_boundary():
    pieces = ["a", " b", " a"]
    idx = _index([0, 1, 2, 0], [0, 0, 0, 1], [0, 1, 2, 0], [0], [1], [2])
    initial, wid = phase9_word_f1.word_ids(idx, pieces)
    assert initial.all()
    assert wid[0] == wid[3]  # "a" at the start of doc 1 is the same word as "a" in doc 0


def test_train_subsets_nested_and_capped_at_available():
    doc = np.repeat(np.arange(10), 100)  # 10 docs x 100 tokens, all train
    idx = _index(list(range(1000)), doc, np.tile(np.arange(100), 10), np.arange(1000), [0], [0])
    subs = phase9_sample_size.train_subsets(idx, (250, 500, 4000), seed=0)
    sizes = sorted(subs)
    assert sizes == [300, 500, 1000]  # whole docs; 4000 capped at the full split, once
    for a, b in zip(sizes, sizes[1:]):
        assert set(subs[a]).issubset(subs[b])


def test_domain_gain_ratio(tmp_path):
    def doc(feature, ce):
        return {"model": "m", "feature": feature, "layers": {"3": {"status": "ok", "metrics": {
            "doc_strata": {"domain": {"code": {"cross_entropy_bits": ce, "n_slots": 8,
                                               "family_a": {"set_agreement@k": 0.5}}}}}}}}
    d = tmp_path / "m"
    d.mkdir()
    (d / "m__F0.json").write_text(json.dumps(doc("F0", 5.0)))
    (d / "m__F1.json").write_text(json.dumps(doc("F1", 4.0)))
    phase9_domain.main(["--src", str(tmp_path), "--out", str(tmp_path / "o")])
    rows = json.loads((tmp_path / "o" / "t9_4_domain.json").read_text())
    (r,) = [x for x in rows if x["feature"] == "F1"]
    assert r["gain_bits"] == pytest.approx(1.0) and r["gain_ratio"] == pytest.approx(0.2)


def test_phase8_conclusions_verdicts(tmp_path):
    def write(d, vals):
        d.mkdir()
        for feat, v in vals.items():
            (d / f"olmoe-0125__{feat}.json").write_text(json.dumps(
                {"layers": {"1": {"status": "ok", "metrics": {"family_b": {"ratio": v}}}}}))
    write(tmp_path / "a", {"F0": 0.0, "F1": 0.2, "F4": 0.3})
    write(tmp_path / "b", {"F0": 0.0, "F1": 0.21, "F4": 0.29})
    out = tmp_path / "r.json"
    rc = phase8_conclusions.main(["--ref", str(tmp_path / "a"), "--cand", str(tmp_path / "b"),
                                  "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["conclusions"] == "unchanged"
    write(tmp_path / "c", {"F0": 0.0, "F1": 0.35, "F4": 0.3})
    rc = phase8_conclusions.main(["--ref", str(tmp_path / "a"), "--cand", str(tmp_path / "c"),
                                  "--out", str(out)])
    assert rc == 1
