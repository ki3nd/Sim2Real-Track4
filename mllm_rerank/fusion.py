"""Top-k gating + Eq.4 fusion of structural (S_str) and semantic (S_sem) scores."""
import torch


def build_topk_and_gate(s_str, top_k, xi):
    k = min(top_k, s_str.size(1))
    topk_idx = s_str.topk(k=k, dim=1).indices
    row_max = s_str.max(dim=1).values
    gate_mask = (row_max > xi).tolist()
    return topk_idx, gate_mask


def fuse_scores(s_str, topk_idx, s_sem, lam):
    s_final = s_str.clone()
    if not s_sem:
        return s_final
    # min-max normalize semantic scores across processed (i,j) entries
    vals = list(s_sem.values())
    if len(vals) >= 2:
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        norm = {kk: (vv - lo) / denom for kk, vv in s_sem.items()}
    else:
        norm = {kk: max(0.0, min(1.0, vv)) for kk, vv in s_sem.items()}
    for (i, local_j), sem in norm.items():
        g = int(topk_idx[i, local_j].item())
        s_final[i, g] = lam * float(s_str[i, g]) + (1 - lam) * float(sem)
    return s_final
