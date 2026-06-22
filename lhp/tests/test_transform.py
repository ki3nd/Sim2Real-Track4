from PIL import Image
import torch
from lhp.transform import LHPTransform


def _img():
    return Image.new("RGB", (640, 480), (123, 117, 104))


def test_output_shape_is_resolution():
    t = LHPTransform(resolution=384)
    out, view = t(_img())
    assert isinstance(out, torch.Tensor) and out.shape == (3, 384, 384)
    assert view in ("local", "global")


def test_local_prob_zero_is_always_global():
    t = LHPTransform(resolution=384, local_prob=0.0)
    assert all(t(_img())[1] == "global" for _ in range(20))


def test_local_prob_one_is_always_local():
    t = LHPTransform(resolution=384, local_prob=1.0)
    assert all(t(_img())[1] == "local" for _ in range(20))
