import torch
from lhp.beit3_loader import build_beit3_retrieval
from lhp.masked_model import MaskedBEiT3ForRetrieval, build_vision_padding_mask


def test_build_returns_masked_subclass_with_cliploss():
    m = build_beit3_retrieval(drop_path_rate=0.0)
    assert isinstance(m, MaskedBEiT3ForRetrieval)
    assert type(m.criterion).__name__ == "ClipLoss"


def test_build_vision_padding_mask_prepends_false_cls_column():
    pm = torch.tensor([[True, False, True, False],
                       [False, False, False, True]])
    vpm = build_vision_padding_mask(pm)
    assert tuple(vpm.shape) == (2, 5)
    assert vpm[:, 0].tolist() == [False, False]   # CLS never masked
    assert torch.equal(vpm[:, 1:], pm)


def test_forward_parity_nomask_vs_allfalse_and_masked_backward():
    torch.manual_seed(0)
    m = build_beit3_retrieval(drop_path_rate=0.0).train()
    img = torch.randn(2, 3, 384, 384)
    txt = torch.randint(5, 64000, (2, 64))
    pad = torch.zeros(2, 64, dtype=torch.long)
    n_tokens = 1 + (384 // 16) ** 2                      # 577

    v_none = m(image=img, only_infer=True)[0]
    allfalse = torch.zeros(2, n_tokens, dtype=torch.bool)
    v_allfalse = m(image=img, vision_padding_mask=allfalse, only_infer=True)[0]
    assert torch.allclose(v_none, v_allfalse, atol=1e-5)  # mask path == parent when all-False

    vpm = torch.zeros(2, n_tokens, dtype=torch.bool); vpm[:, 1:300] = True   # mask some patches
    loss, _, _ = m(image=img, text_description=txt, padding_mask=pad, vision_padding_mask=vpm)
    assert loss.ndim == 0
    loss.backward()
