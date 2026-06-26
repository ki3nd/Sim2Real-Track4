import torch
from models.masked_swin import MaskedSwin, generate_mim_mask

SWIN = dict(img_size=224, patch_size=4, in_chans=3, embed_dim=128,
            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7,
            drop_path_rate=0.0)


def _build():
    torch.manual_seed(0)
    return MaskedSwin(**SWIN).eval()


def test_parity_with_no_mask():
    m = _build()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_none = m(x)
        out_nonearg = m(x, mask=None)
    assert out_none.shape == (2, 1 + 49, 1024)
    assert torch.allclose(out_none, out_nonearg, atol=1e-6)   # None path == parent path


def test_masked_changes_output():
    m = _build()
    x = torch.randn(2, 3, 224, 224)
    patch_mask, _ = generate_mim_mask(2, 0.6, x.device)
    with torch.no_grad():
        out = m(x, mask=patch_mask)
    assert out.shape == (2, 1 + 49, 1024)
    with torch.no_grad():
        assert not torch.allclose(out, m(x))                  # masking alters the output


def test_generate_mim_mask_shapes_and_ratio():
    patch_mask, cell_mask = generate_mim_mask(4, 0.6, torch.device("cpu"))
    assert patch_mask.shape == (4, 56 * 56) and patch_mask.dtype == torch.bool
    assert cell_mask.shape == (4, 7, 7)
    # 0.6 of 49 cells masked == round(29.4)=29 cells -> each cell = 8x8 patches
    assert int(cell_mask[0].sum()) == round(0.6 * 49)
    assert int(patch_mask[0].sum()) == round(0.6 * 49) * 64
