from mllm_rerank.cgcr_prompts import (
    DECOMPOSER_PROMPT, VERIFIER_PROMPT, CRITIC_PROMPT, ADJUDICATOR_PROMPT,
    parse_claims, parse_verdicts, parse_adjudication,
)


def test_prompt_fields():
    assert "{query}" in DECOMPOSER_PROMPT
    assert "{claims}" in VERIFIER_PROMPT
    assert "{claims}" in CRITIC_PROMPT
    assert "{query}" in ADJUDICATOR_PROMPT and "{n}" in ADJUDICATOR_PROMPT


def test_parse_claims_valid():
    raw = '{"claims":[{"id":1,"category":"action","text":"falling down"},{"id":2,"category":"upper","text":"red shirt"}]}'
    out = parse_claims(raw)
    assert len(out) == 2
    assert out[0] == {"id": 1, "category": "action", "text": "falling down"}
    assert out[1]["category"] == "upper"


def test_parse_claims_assigns_sequential_ids_when_missing():
    raw = '{"claims":[{"category":"gender","text":"male"},{"category":"action","text":"running"}]}'
    out = parse_claims(raw)
    assert [c["id"] for c in out] == [1, 2]


def test_parse_claims_bad_json_returns_empty():
    assert parse_claims("not json at all") == []


def test_parse_verdicts_normalizes_label_and_clamps_conf():
    raw = '{"verdicts":[{"id":1,"label":"entailed","conf":1.4,"evidence":"x"},{"id":2,"label":"weird","conf":-0.2,"evidence":"y"}]}'
    out = parse_verdicts(raw)
    assert out[0]["label"] == "ENTAILED" and out[0]["conf"] == 1.0
    assert out[1]["label"] == "NEUTRAL" and out[1]["conf"] == 0.0


def test_parse_verdicts_embedded_json():
    raw = 'Here:\n{"verdicts":[{"id":3,"label":"CONTRADICTED","conf":0.7,"evidence":"push-ups"}]} done'
    out = parse_verdicts(raw)
    assert len(out) == 1 and out[0]["id"] == 3 and out[0]["label"] == "CONTRADICTED"


def test_parse_adjudication_valid_permutation():
    assert parse_adjudication('{"ranking":[2,0,1]}', 3) == [2, 0, 1]


def test_parse_adjudication_fallback_identity():
    assert parse_adjudication("garbage", 3) == [0, 1, 2]
    # incomplete / out-of-range permutation -> identity
    assert parse_adjudication('{"ranking":[0,0,5]}', 3) == [0, 1, 2]
