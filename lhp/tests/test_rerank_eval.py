import torch
import pytest
from lhp.rerank_eval import eval_args, assert_aligned
from lhp.infer import similarity


def test_eval_args_distributed_false():
    assert eval_args().distributed is False


def test_assert_aligned_passes_then_raises():
    sims = torch.zeros(3, 5)          # [N_query=3, N_gallery=5]
    assert_aligned(sims, 3, 5)        # correct -> no raise
    with pytest.raises(AssertionError):
        assert_aligned(sims, 5, 3)    # swapped -> raise


def test_similarity_is_query_by_gallery():
    img = torch.randn(5, 8)           # 5 gallery
    txt = torch.randn(3, 8)           # 3 queries
    s = similarity(img, txt)          # txt @ img.T
    assert tuple(s.shape) == (3, 5)   # [N_query, N_gallery]
