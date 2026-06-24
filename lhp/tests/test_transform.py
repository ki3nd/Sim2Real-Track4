from PIL import Image
import torch
from lhp.transform import LHPTransform


def _img():
    return Image.new("RGB", (640, 480), (123, 117, 104))


def test_output_shape_is_resolution():
    t = LHPTransform(resolution=384)
    out, view, _ = t(_img())
    assert isinstance(out, torch.Tensor) and out.shape == (3, 384, 384)
    assert view in ("local", "global", "masked")


def test_local_prob_zero_is_always_global():
    t = LHPTransform(resolution=384, local_prob=0.0)
    assert all(t(_img())[1] == "global" for _ in range(20))


def test_local_prob_one_is_always_local():
    t = LHPTransform(resolution=384, local_prob=1.0)
    assert all(t(_img())[1] == "local" for _ in range(20))


def test_returns_three_tuple_with_patch_mask():
    t = LHPTransform(resolution=384, local_prob=1.0)   # always local
    out, view, patch_mask = t(_img())
    assert tuple(out.shape) == (3, 384, 384) and view == "local"
    assert patch_mask.dtype == torch.bool and patch_mask.shape == (576,)
    assert patch_mask.sum().item() == 0                # local -> nothing masked


def test_masked_view_masks_40_to_60_percent():
    t = LHPTransform(resolution=384, local_prob=0.0, masked_prob=1.0)  # always masked
    for _ in range(10):
        out, view, patch_mask = t(_img())
        assert view == "masked" and tuple(out.shape) == (3, 384, 384)
        frac = patch_mask.float().mean().item()
        assert 0.40 - 1e-6 <= frac <= 0.60 + 1e-6      # ratio in [0.4, 0.6]


def test_masked_prob_zero_never_masks():
    t = LHPTransform(resolution=384, local_prob=0.0, masked_prob=0.0)  # always global
    views = [t(_img())[1] for _ in range(20)]
    assert set(views) == {"global"}
