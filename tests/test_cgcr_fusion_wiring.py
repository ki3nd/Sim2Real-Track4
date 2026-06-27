import torch
from mllm_rerank.cgcr import build_sem_dict
from mllm_rerank.fusion import fuse_scores


def test_build_sem_dict_passthrough():
    g = {(0, 1): 0.9, (0, 0): -0.2}
    out = build_sem_dict(g)
    assert out == g          # shape compatible with fuse_scores' s_sem


def test_cgcr_grounded_score_fuses_like_rerank():
    # grounded score promotes candidate at local_j=1 (gallery idx 2)
    s_str = torch.tensor([[0.8, 0.2, 0.5, 0.1]])
    topk_idx = torch.tensor([[0, 2]])
    g = {(0, 0): 0.0, (0, 1): 1.0}        # min-max over {0,1} -> {0,1}
    s_final = fuse_scores(s_str, topk_idx, build_sem_dict(g), lam=0.4)
    # local0->gallery0: 0.4*0.8+0.6*0 = 0.32 ; local1->gallery2: 0.4*0.5+0.6*1 = 0.80
    assert abs(s_final[0, 0].item() - 0.32) < 1e-5
    assert abs(s_final[0, 2].item() - 0.80) < 1e-5
    assert abs(s_final[0, 1].item() - 0.2) < 1e-5 and abs(s_final[0, 3].item() - 0.1) < 1e-5
