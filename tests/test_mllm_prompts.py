from mllm_rerank.prompts import (
    DETECTIVE_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    process_checklist,
    parse_writer_caption,
)


def test_prompts_have_expected_fields():
    assert "{cap}" in DETECTIVE_PROMPT
    assert "{cap}" in WRITER_PROMPT
    assert "{cap}" not in ANALYST_PROMPT  # Analyst takes only the image
    assert "Yes" in DETECTIVE_PROMPT and "No" in DETECTIVE_PROMPT


def test_process_checklist_cleans_indexed_answers():
    raw = (
        "1.Yes, the person is male.\n"
        "2.The person has short black hair.\n"
        "3.He is wearing a red t-shirt.\n"
    )
    out = process_checklist([raw])
    assert len(out) == 1
    sents = out[0]
    # index "N." stripped, "Yes, " stripped, capitalized, trailing period
    assert "The person is male." in sents
    assert "The person has short black hair." in sents
    assert all(s == "" or s.endswith(".") for s in sents)
    # no leading numeric index survives
    assert not any(s[:2] in ("1.", "2.", "3.") for s in sents if s)


def test_process_checklist_drops_colon_header_lines():
    raw = "Answers:\n1.The person is female.\n"
    out = process_checklist([raw])
    assert "The person is female." in out[0]
    assert "Answers:" not in out[0]


def test_parse_writer_caption_valid_json():
    raw = '{"caption": "A man in a red shirt is falling down."}'
    assert parse_writer_caption(raw) == "A man in a red shirt is falling down."


def test_parse_writer_caption_falls_back_on_bad_json():
    raw = "A woman walking a dog."
    assert parse_writer_caption(raw) == "A woman walking a dog."


def test_parse_writer_caption_extracts_json_embedded_in_text():
    raw = 'Output: {"caption": "Two people fighting."} done'
    assert parse_writer_caption(raw) == "Two people fighting."
