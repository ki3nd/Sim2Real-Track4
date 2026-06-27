import torch
from mllm_rerank.fusion import build_topk_and_gate, fuse_scores


def test_build_topk_and_gate():
    s = torch.tensor([[0.9, 0.1, 0.5], [0.05, 0.02, 0.03]])
    topk_idx, gate = build_topk_and_gate(s, top_k=2, xi=0.1)
    assert topk_idx.shape == (2, 2)
    assert topk_idx[0].tolist() == [0, 2]      # 0.9, 0.5
    assert gate == [True, False]               # row1 max 0.9 > 0.1; row2 max 0.05 <= 0.1


def test_fuse_only_touches_processed_positions():
    s = torch.tensor([[0.8, 0.2, 0.5, 0.1]])
    topk_idx = torch.tensor([[0, 2]])          # candidates at gallery idx 0 and 2
    # semantic score only for local_j=1 (gallery idx 2); two entries to enable min-max
    s_sem = {(0, 0): 0.0, (0, 1): 1.0}
    out = fuse_scores(s, topk_idx, s_sem, lam=0.4)
    # local 0 -> gallery 0: 0.4*0.8 + 0.6*0.0 = 0.32
    assert abs(out[0, 0].item() - 0.32) < 1e-5
    # local 1 -> gallery 2: 0.4*0.5 + 0.6*1.0 = 0.80
    assert abs(out[0, 2].item() - 0.80) < 1e-5
    # untouched positions keep s_str
    assert abs(out[0, 1].item() - 0.2) < 1e-5
    assert abs(out[0, 3].item() - 0.1) < 1e-5


def test_fuse_promotes_high_semantic_candidate_in_ranking():
    s = torch.tensor([[0.8, 0.79]])            # idx0 slightly ahead structurally
    topk_idx = torch.tensor([[0, 1]])
    s_sem = {(0, 0): 0.0, (0, 1): 1.0}         # idx1 strongly favored semantically
    out = fuse_scores(s, topk_idx, s_sem, lam=0.4)
    assert out[0, 1] > out[0, 0]               # idx1 now ranks first
