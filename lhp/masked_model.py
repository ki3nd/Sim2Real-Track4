import torch
import torch.nn.functional as F

from lhp.beit3.modeling_finetune import BEiT3ForRetrieval


def build_vision_padding_mask(patch_mask):
    """patch_mask: bool [B, num_patches] -> vision key-padding mask [B, 1+num_patches]
    with the CLS column (0) always False (never masked)."""
    cls_col = torch.zeros(patch_mask.size(0), 1, dtype=torch.bool, device=patch_mask.device)
    return torch.cat([cls_col, patch_mask], dim=1)


class MaskedBEiT3ForRetrieval(BEiT3ForRetrieval):
    """BEiT3ForRetrieval + an optional per-sample vision key-padding mask.

    vision_padding_mask: bool [B, 1+num_patches], True = patch blocked from attention
    (and its input embedding zeroed). vision_padding_mask=None reproduces the parent
    forward exactly."""

    def forward(self, image=None, text_description=None, padding_mask=None,
                vision_padding_mask=None, only_infer=False, **kwargs):
        if image is not None and vision_padding_mask is not None:
            x = self.beit3.vision_embed(image)
            x = self.beit3.encoder(
                src_tokens=None,
                encoder_padding_mask=vision_padding_mask,
                token_embeddings=x,
                multiway_split_position=-1,
            )["encoder_out"]
            vision_cls = F.normalize(self.vision_head(x[:, 0, :]), dim=-1)
        elif image is not None:
            x = self.beit3(textual_tokens=None, visual_tokens=image,
                           text_padding_position=None)["encoder_out"]
            vision_cls = F.normalize(self.vision_head(x[:, 0, :]), dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            x = self.beit3(textual_tokens=text_description, visual_tokens=None,
                           text_padding_position=padding_mask)["encoder_out"]
            language_cls = F.normalize(self.language_head(x[:, 0, :]), dim=-1)
        else:
            language_cls = None

        if only_infer:
            return vision_cls, language_cls
        loss, logits_per_image, logits_per_text = self.criterion(
            vision_cls, language_cls, self.logit_scale.exp())
        return loss, vision_cls, language_cls
