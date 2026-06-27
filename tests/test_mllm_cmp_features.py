import torch
import torch.nn.functional as F
from mllm_rerank.cmp_features import embed_texts


class _FakeTok:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        n = len(texts)

        class _Enc:
            input_ids = torch.zeros(n, max_length, dtype=torch.long)
            attention_mask = torch.ones(n, max_length, dtype=torch.long)

            def to(self, device):
                return self
        return _Enc()


class _FakeModel:
    """Mimics Search.get_text_embeds / get_text_feat: maps each row to a fixed direction."""
    def get_text_embeds(self, input_ids, attention_mask):
        b = input_ids.size(0)
        return torch.arange(1, b + 1, dtype=torch.float).view(b, 1, 1).repeat(1, 4, 8)

    def get_text_feat(self, text_embed):
        # use the [:,0] token -> [b, 8]
        return text_embed[:, 0, :]


def test_embed_texts_returns_l2_normalized_rows():
    model, tok = _FakeModel(), _FakeTok()
    cfg = {"max_tokens": 56}
    out = embed_texts(model, tok, ["a", "b", "c"], device=torch.device("cpu"), config=cfg)
    assert out.shape == (3, 8)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)  # L2-normalized
