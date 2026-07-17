from finrag.eval.judge import parse_verdict
from finrag.retrieve import rrf_fuse


def test_rrf_prefers_agreement():
    dense = ["a", "b", "c", "d"]
    bm25 = ["c", "a", "e"]
    fused = rrf_fuse([dense, bm25], k=60)
    assert fused[0] == "a"  # rank 1 + rank 2 beats everything
    assert fused.index("c") < fused.index("b")  # agreement beats single-list rank


def test_rrf_single_ranking_is_identity():
    assert rrf_fuse([["x", "y", "z"]], k=60) == ["x", "y", "z"]


def test_parse_verdict_json():
    v, r = parse_verdict('{"verdict": "correct", "reason": "matches"}')
    assert v == "correct" and r == "matches"


def test_parse_verdict_json_embedded_in_prose():
    v, _ = parse_verdict('Sure! Here is my grade:\n{"verdict": "refusal", "reason": "declined"}')
    assert v == "refusal"


def test_parse_verdict_fallback_word():
    v, _ = parse_verdict("The answer is incorrect because the year is wrong.")
    assert v == "incorrect"


def test_parse_verdict_garbage_defaults_incorrect():
    v, _ = parse_verdict("¯\\_(ツ)_/¯")
    assert v == "incorrect"
