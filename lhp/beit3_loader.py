from lhp.beit3.modeling_utils import _get_base_config
from lhp.beit3 import utils as beit3_utils
from lhp.masked_model import MaskedBEiT3ForRetrieval


def build_beit3_retrieval(drop_path_rate: float = 0.1):
    """Build MaskedBEiT3ForRetrieval (base, patch16, 384). Distributed must be
    initialized BEFORE calling so the internal ClipLoss gets correct rank/world_size.
    With vision_padding_mask=None it behaves identically to BEiT3ForRetrieval."""
    args = _get_base_config(img_size=384, drop_path_rate=drop_path_rate)
    return MaskedBEiT3ForRetrieval(args)


def load_pretrained(model, ckpt_path: str):
    """Load a BeiT-3 retrieval checkpoint with positional-embedding interpolation."""
    beit3_utils.load_model_and_may_interpolate(
        ckpt_path, model, model_key="model|module", model_prefix="")
    return model
