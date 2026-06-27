from mllm_rerank.claims import category_weight, aggregate_self_consistency, score_ground

TABLE = {"action": 1.5, "upper": 1.0, "gender": 0.8, "background": 0.3}
DEFAULT = 0.7


def test_category_weight_lookup_and_default():
    assert category_weight("action", TABLE, DEFAULT) == 1.5
    assert category_weight("ACTION", TABLE, DEFAULT) == 1.5
    assert category_weight("unknown", TABLE, DEFAULT) == DEFAULT


def test_aggregate_majority_label_and_mean_conf():
    samples = [
        [{"id": 1, "label": "ENTAILED", "conf": 0.8}],
        [{"id": 1, "label": "ENTAILED", "conf": 0.6}],
        [{"id": 1, "label": "CONTRADICTED", "conf": 0.9}],
    ]
    out = aggregate_self_consistency(samples)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["label"] == "ENTAILED"
    assert abs(out[0]["conf"] - 0.7) < 1e-6   # mean of the two ENTAILED confs


def test_aggregate_label_tie_prefers_neutral():
    samples = [
        [{"id": 5, "label": "ENTAILED", "conf": 0.9}],
        [{"id": 5, "label": "CONTRADICTED", "conf": 0.9}],
    ]
    out = aggregate_self_consistency(samples)
    assert out[0]["label"] == "NEUTRAL"


def test_score_ground_signs_and_weights():
    claims = [
        {"id": 1, "category": "action", "text": "falling"},
        {"id": 2, "category": "upper", "text": "red shirt"},
        {"id": 3, "category": "gender", "text": "male"},
    ]
    verdicts = [
        {"id": 1, "label": "CONTRADICTED", "conf": 1.0},  # -1.5 weight contribution
        {"id": 2, "label": "ENTAILED", "conf": 1.0},      # +1.0
        {"id": 3, "label": "NEUTRAL", "conf": 0.5},       # 0
    ]
    s = score_ground(claims, verdicts, TABLE, DEFAULT)
    # (1.5*-1 + 1.0*1 + 0.8*0) / (1.5+1.0+0.8) = -0.5/3.3
    assert abs(s - (-0.5 / 3.3)) < 1e-6


def test_score_ground_all_neutral_is_zero():
    claims = [{"id": 1, "category": "action", "text": "x"}]
    verdicts = [{"id": 1, "label": "NEUTRAL", "conf": 0.9}]
    assert score_ground(claims, verdicts, TABLE, DEFAULT) == 0.0


def test_score_ground_missing_verdict_treated_neutral():
    claims = [{"id": 1, "category": "action", "text": "x"}]
    assert score_ground(claims, [], TABLE, DEFAULT) == 0.0
