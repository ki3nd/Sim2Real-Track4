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
    # eval.mAP reads cmc[0],[4],[9] (R@1/5/10) -> needs >=10 gallery items.
    # Perfect ranking: each query's own image scores highest (diagonal).
    n = 10
    scores = (np.eye(n, dtype="float32") * 0.8) + 0.1
    res = mAP(scores, g_pids=list(range(n)), q_pids=list(range(n)))
    assert round(float(res["R1"])) == 100
    assert round(float(res["mAP"])) == 100
