from mllm_rerank.cgcr_squad import (
    decompose, verify_pairs, should_continue_recovery,
)


class _LLM:
    """Decomposer (text) returns one claim; Verifier (image) returns ENTAILED for image=='match' else CONTRADICTED."""
    def generate_response_text(self, questions, t=0.01):
        return ['{"claims":[{"id":1,"category":"action","text":"falling"}]}' for _ in questions]

    def generate_response_multi_images(self, questions, images, t=0.01):
        out = []
        for img in images:
            if img == "match":
                out.append('{"verdicts":[{"id":1,"label":"ENTAILED","conf":0.9,"evidence":"e"}]}')
            else:
                out.append('{"verdicts":[{"id":1,"label":"CONTRADICTED","conf":0.8,"evidence":"e"}]}')
        return out


def test_decompose_returns_claims_per_query():
    out = decompose(_LLM(), ["q1", "q2"])
    assert len(out) == 2
    assert out[0][0]["category"] == "action"


def test_verify_pairs_aggregates_self_consistency():
    claims = [{"id": 1, "category": "action", "text": "falling"}]
    work = [(claims, "match"), (claims, "other")]
    out = verify_pairs(_LLM(), work, n_samples=3)
    assert len(out) == 2
    assert out[0][0]["label"] == "ENTAILED"     # 3/3 ENTAILED
    assert out[1][0]["label"] == "CONTRADICTED"  # 3/3 CONTRADICTED


def test_verify_pairs_empty():
    assert verify_pairs(_LLM(), [], n_samples=3) == []


def test_should_continue_stops_on_hit():
    assert should_continue_recovery(0.7, 5, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) is None


def test_should_continue_deepens_when_below_hit():
    assert should_continue_recovery(0.3, 5, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) == 10


def test_should_continue_stops_at_kmax():
    assert should_continue_recovery(0.3, 20, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=5) is None


def test_should_continue_stops_at_max_rounds():
    assert should_continue_recovery(0.3, 5, 1, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) is None
