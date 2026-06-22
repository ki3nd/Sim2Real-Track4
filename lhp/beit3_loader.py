from lhp.beit3.modeling_finetune import beit3_base_patch16_384_retrieval
from lhp.beit3 import utils as beit3_utils


def build_beit3_retrieval(drop_path_rate: float = 0.1):
    """Build BEiT3ForRetrieval (base, patch16, 384). Distributed must be initialized
    BEFORE calling this so the internal ClipLoss gets correct rank/world_size."""
    return beit3_base_patch16_384_retrieval(pretrained=False, drop_path_rate=drop_path_rate)


def load_pretrained(model, ckpt_path: str):
    """Load a BeiT-3 retrieval checkpoint with positional-embedding interpolation."""
    beit3_utils.load_model_and_may_interpolate(
        ckpt_path, model, model_key="model|module", model_prefix="")
    return model
