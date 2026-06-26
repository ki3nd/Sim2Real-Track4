import torch
from uit.xvlm_init import remap_xvlm_state_dict


def test_remap_renames_vision_encoder_prefix_only():
    sd = {
        "vision_encoder.patch_embed.proj.weight": torch.zeros(1),
        "vision_encoder.layers.0.x": torch.zeros(1),
        "text_encoder.bert.x": torch.zeros(1),
        "vision_proj.weight": torch.zeros(1),
        "text_proj.weight": torch.zeros(1),
        "itm_head.weight": torch.zeros(1),
        "temp": torch.zeros(1),
    }
    out = remap_xvlm_state_dict(sd)
    assert "vision.patch_embed.proj.weight" in out
    assert "vision.layers.0.x" in out
    assert not any(k.startswith("vision_encoder.") for k in out)   # all renamed
    # non-vision keys are kept verbatim
    for k in ["text_encoder.bert.x", "vision_proj.weight", "text_proj.weight", "itm_head.weight", "temp"]:
        assert k in out
