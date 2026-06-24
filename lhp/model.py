import torch
import torch.nn as nn
from lhp.beit3_loader import build_beit3_retrieval, load_pretrained


class LHPRetriever(nn.Module):
    """BeiT-3 base/384 retrieval + built-in ClipLoss. Build AFTER distributed init."""

    def __init__(self, ckpt_path=None, drop_path_rate=0.1):
        super().__init__()
        self.beit3 = build_beit3_retrieval(drop_path_rate=drop_path_rate)
        if ckpt_path:
            load_pretrained(self.beit3, ckpt_path)

    def forward(self, image, text_ids, padding_mask, vision_padding_mask=None):
        # BEiT3ForRetrieval computes ClipLoss internally over in-batch (gathered) negatives
        return self.beit3(image=image, text_description=text_ids, padding_mask=padding_mask,
                          vision_padding_mask=vision_padding_mask)

    @torch.no_grad()
    def encode_image(self, image):
        vision_cls, _ = self.beit3(image=image, only_infer=True)
        return vision_cls

    @torch.no_grad()
    def encode_text(self, text_ids, padding_mask):
        _, language_cls = self.beit3(text_description=text_ids, padding_mask=padding_mask, only_infer=True)
        return language_cls
