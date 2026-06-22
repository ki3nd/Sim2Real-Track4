import torch.nn as nn
from lhp.beit3_loader import build_beit3_retrieval, beit3_utils


def test_build_beit3_retrieval_returns_module_with_cliploss():
    model = build_beit3_retrieval(drop_path_rate=0.0)
    assert isinstance(model, nn.Module)
    # ClipLoss is the contrastive criterion baked into BEiT3ForRetrieval
    assert type(model.criterion).__name__ == "ClipLoss"


def test_trimmed_utils_has_no_broken_imports():
    # importing beit3_utils must succeed despite upstream torch._six/tensorboardX
    assert hasattr(beit3_utils, "load_model_and_may_interpolate")
    assert hasattr(beit3_utils, "ClipLoss")
