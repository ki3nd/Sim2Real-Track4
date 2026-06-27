import torch
from mllm_rerank.metrics import rank, get_metrics


def test_perfect_ranking_gives_r1_100():
    # 3 queries, 3 gallery; identity => each query's match is rank-1
    sim = torch.eye(3)
    qids = torch.tensor([0, 1, 2])
    gids = torch.tensor([0, 1, 2])
    all_cmc, mAP, mINP, indices = rank(sim, qids, gids, max_rank=3, get_mAP=True)
    assert float(all_cmc[0]) == 100.0
    assert float(mAP) == 100.0


def test_get_metrics_row_shape_and_r1():
    sim = torch.eye(10)
    qids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    gids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    row = get_metrics(sim, qids, gids, "unit-t2i", retur_indices=False)
    assert len(row) == 7
    assert row[0] == "unit-t2i"
    assert round(float(row[1]), 2) == 100.00  # R1
    # rSum == R1+R5+R10
    assert round(float(row[6]), 2) == round(float(row[1]) + float(row[2]) + float(row[3]), 2)


def test_worst_ranking_lowers_r1():
    # anti-diagonal: correct match is always last => R1 should be 0
    sim = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    qids = torch.tensor([0, 1])
    gids = torch.tensor([0, 1])
    all_cmc, mAP, mINP, indices = rank(sim, qids, gids, max_rank=2, get_mAP=True)
    assert float(all_cmc[0]) == 0.0
