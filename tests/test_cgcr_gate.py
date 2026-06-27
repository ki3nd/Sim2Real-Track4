import torch
from mllm_rerank.cgcr_gate import ambiguity_gate


def test_peaked_row_gated_out_flat_row_activated():
    s = torch.tensor([
        [0.9, 0.1, 0.05],   # margin 0.8 -> not ambiguous -> skip (False)
        [0.50, 0.49, 0.2],  # margin 0.01 -> ambiguous   -> activate (True)
    ])
    mask, margins = ambiguity_gate(s, gate_margin=0.05)
    assert mask == [False, True]
    assert abs(margins[0] - 0.8) < 1e-6
    assert abs(margins[1] - 0.01) < 1e-6


def test_boundary_is_strict_less_than():
    s = torch.tensor([[0.5, 0.45]])   # margin exactly 0.05
    mask, _ = ambiguity_gate(s, gate_margin=0.05)
    assert mask == [False]            # 0.05 < 0.05 is False -> skip
