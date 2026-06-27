"""Ambiguity gate: activate the MLLM rerank only when Stage-1 is undecided (small top1-top2 margin)."""
import torch


def ambiguity_gate(s_str, gate_margin):
    mask, margins = [], []
    for i in range(s_str.size(0)):
        row = s_str[i]
        k = min(2, row.size(0))
        top = torch.topk(row, k=k).values
        if k >= 2:
            margin = float(top[0] - top[1])
        else:
            margin = float(top[0])
        margins.append(margin)
        mask.append(margin < gate_margin)
    return mask, margins
