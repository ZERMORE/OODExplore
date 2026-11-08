"""Server-side protocol tests. No model download or CUDA needed."""
import numpy as np
import pytest

from ood_explore.analysis import cluster_indices, evaluate, fit_reference, risks
from ood_explore.pipeline import paired_features, parse_answer, check_replay
from ood_explore.data import prompt


def example(i, correct=True, cluster=None):
    return {"sample_id": str(i), "image_id": str(i if cluster is None else cluster),
            "is_correct": correct, "eligible": True, "parsed_answer": 0,
            "stop_reason": "eos", "features": {"nci": float(i), "nci_gain": float(i),
            "fdbd": float(i), "fdbd_gain": float(i), "logprob": float(i),
            "logprob_gain": float(i), "negative_entropy": float(i)}}


def test_reference_is_frozen_and_direction_is_risk():
    reference = fit_reference([example(0), example(1)])
    saved = {k: list(v) for k, v in reference.items()}
    assert risks(example(-1)["features"], reference)["joint_fdbd"] == 1
    assert risks(example(2)["features"], reference)["joint_fdbd"] == 0
    assert risks(example(0)["features"], reference)["joint_fdbd"] == .5
    assert reference == saved


def test_error_positive_and_exclusions():
    rows = [example(0, False), example(1, False), example(2), example(3)]
    excluded = example(9, False)
    excluded["eligible"], excluded["parsed_answer"] = False, None
    report = evaluate(rows+[excluded], fit_reference(rows), bootstrap=20)
    assert report["metrics"]["joint_fdbd"]["auroc"] == 1
    assert report["n_eligible"] == 4
    assert report["parse_failures"] == 1
    assert report["accuracy_all"] == .4
    assert report["paired_delta_auroc"]["joint_fdbd_minus_fdbd"]["ci95"] == [0, 0]


def test_cluster_resampling_keeps_same_image_questions_together():
    rows = [example(0, cluster=0), example(1, cluster=0), example(2, cluster=1)]
    for seed in range(10):
        ids = cluster_indices(rows, np.random.default_rng(seed)).tolist()
        assert ids.count(0) == ids.count(1)


def test_one_class_is_explicitly_undefined():
    rows = [example(0), example(1)]
    assert "undefined_reason" in evaluate(rows, fit_reference(rows))


def test_answer_parser_does_not_take_reasoning_letters():
    assert parse_answer("A seems wrong.\nFinal answer: B.", 3) == 1
    assert parse_answer("Final answer: D", 3) is None
    assert parse_answer("Final answer: A\nFinal answer: B", 3) is None
    assert parse_answer("The answer might be A", 3) is None


def test_prompt_never_contains_gold_explanations():
    text = prompt({"question": "Question", "choices": ["X", "Y"],
                   "answer": 1, "lecture": "LEAK", "solution": "LEAK"})
    assert "LEAK" not in text and "A. X" in text and "B. Y" in text


def test_pair_alignment_and_visual_gain():
    def trace(token, nci):
        return {"trace": [{"position": 0, "token_id": token, "scores": {
            "nci": nci, "fdbd_k1000": nci, "mean_logprob": -2., "negative_entropy": -1.}}]}
    means, records = paired_features(trace(1, 3), trace(1, 1), 1000)
    assert means["fdbd_gain"] == 2 and records[0]["position"] == 0
    with pytest.raises(ValueError, match="identical tokens"):
        paired_features(trace(1, 3), trace(2, 1), 1000)


def test_replay_verification_detects_shifted_scores():
    a = {"token_ids": [1], "trace": [{"position": 0, "token_id": 1, "scores": {"nci": 1.}}]}
    b = {"token_ids": [1], "trace": [{"position": 0, "token_id": 1, "scores": {"nci": 2.}}]}
    with pytest.raises(RuntimeError, match="mismatch"):
        check_replay(a, b, .001, .001)
