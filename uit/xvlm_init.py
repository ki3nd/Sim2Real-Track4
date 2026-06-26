import torch


def remap_xvlm_state_dict(sd):
    """Rename X-VLM keys to UIT module names: vision_encoder.* -> vision.* ;
    text_encoder.*/vision_proj.*/text_proj.*/itm_head.*/temp kept verbatim."""
    out = {}
    for k, v in sd.items():
        if k.startswith("vision_encoder."):
            out["vision." + k[len("vision_encoder."):]] = v
        else:
            out[k] = v
    return out


def load_xvlm(model, ckpt_path):
    """Load an X-VLM checkpoint into a UIT model (strict=False). mim_decoder + mask_token
    are not in X-VLM and stay randomly initialized. Returns (missing_keys, unexpected_keys)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = remap_xvlm_state_dict(sd)
    msg = model.load_state_dict(sd, strict=False)
    print("xvlm_init missing:", [k for k in msg.missing_keys][:20])
    print("xvlm_init unexpected:", [k for k in msg.unexpected_keys][:20])
    return msg.missing_keys, msg.unexpected_keys
