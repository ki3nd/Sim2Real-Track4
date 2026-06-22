from lhp.tokenization import tokenize_caption


class StubTok:
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    def tokenize(self, text):
        return text.split()
    def convert_tokens_to_ids(self, toks):
        return [10 + i for i, _ in enumerate(toks)]


def test_pads_to_max_len_and_marks_padding():
    ids, mask = tokenize_caption(StubTok(), "a b c", max_len=8)
    assert len(ids) == 8 and len(mask) == 8
    assert ids[0] == 0 and ids[4] == 2          # bos ... eos  (3 words + bos + eos = 5 real)
    assert ids[5:] == [1, 1, 1]                 # pad
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1]


def test_truncates_long_text():
    ids, mask = tokenize_caption(StubTok(), " ".join(["w"] * 100), max_len=8)
    assert len(ids) == 8
    assert ids[0] == 0 and ids[7] == 2          # bos at 0, eos at last, 6 content
    assert mask == [0] * 8
