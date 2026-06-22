import json, os
from PIL import Image
from lhp.transform import LHPTransform
from lhp.dataset import LHPDataset


def _setup(tmp_path):
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    Image.new("RGB", (320, 240), (100, 110, 120)).save(root / "train" / "0.jpg")
    ann = tmp_path / "ann.jsonl"
    with open(ann, "w") as f:
        f.write(json.dumps({"image": "train/0.jpg",
                            "caption": "a person is running on grass"}) + "\n")
    return str(ann), str(root)


def test_returns_image_caption_path(tmp_path):
    ann, root = _setup(tmp_path)
    ds = LHPDataset([ann], root, LHPTransform(resolution=384), eda=False)
    assert len(ds) == 1
    img, cap, path = ds[0]
    assert tuple(img.shape) == (3, 384, 384)
    assert isinstance(cap, str) and len(cap) > 0
    assert path.endswith("train/0.jpg")
