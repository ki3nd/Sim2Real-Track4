import numpy as np
from lhp.eval import build_test_index
from eval import mAP


def test_build_test_index_flattens_captions_and_ids():
    ann = [
        {"image": "a.jpg", "image_id": 10, "caption": ["a person running", "a man runs"]},
        {"image": "b.jpg", "image_id": 20, "caption": ["a person sitting"]},
    ]
    g_pids, captions, q_pids = build_test_index(ann, max_words=56)
    assert g_pids == [10, 20]              # one per gallery image
    assert len(captions) == 3 and all(isinstance(c, str) for c in captions)
    assert q_pids == [10, 10, 20]          # one per caption, owning image's id


def test_map_perfect_ranking_is_r1_100():
    # 2 gallery (ids 10,20), 2 queries (ids 10,20); each query scores its own image highest
    scores = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    res = mAP(scores, g_pids=[10, 20], q_pids=[10, 20])
    assert round(float(res["R1"])) == 100
    assert round(float(res["mAP"])) == 100
