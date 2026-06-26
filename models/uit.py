import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMDecoder(nn.Module):
    """SimMIM lightweight decoder: linear-predict the raw 32x32x3 pixel block for each of the
    49 final-stage tokens, then fold into a [B, 3, 224, 224] image."""

    def __init__(self, in_dim=1024, mask_patch_size=32):
        super().__init__()
        self.p = mask_patch_size
        self.grid = 224 // mask_patch_size                 # 7
        self.head = nn.Linear(in_dim, mask_patch_size * mask_patch_size * 3)

    def forward(self, spatial):                            # [B, 49, in_dim]
        b = spatial.size(0)
        x = self.head(spatial)                             # [B, 49, p*p*3]
        x = x.view(b, self.grid, self.grid, 3, self.p, self.p)        # [B,7,7,3,32,32]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()                 # [B,3,7,32,7,32]
        return x.view(b, 3, self.grid * self.p, self.grid * self.p)  # [B,3,224,224]


def mim_loss(recon, image, cell_mask):
    """L1 over masked 32x32 cells only. cell_mask: bool [B, 7, 7]."""
    p = recon.size(-1) // cell_mask.size(-1)               # 32
    pixel_mask = cell_mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)  # [B,224,224]
    pixel_mask = pixel_mask.unsqueeze(1)                   # [B,1,224,224]
    diff = (recon - image).abs() * pixel_mask
    denom = pixel_mask.sum() * recon.size(1) + 1e-6        # masked pixels * channels
    return diff.sum() / denom


from models.masked_swin import MaskedSwin, generate_mim_mask
from models.bert import BertConfig, BertForMaskedLM


class UIT(nn.Module):
    """Unified Image-Text model: ITC + ITM + MLM + MIM on a single shared MaskedSwin encoder.
    Standalone (no CMP import); no pose; no IHNM."""

    def __init__(self, config):
        super().__init__()
        s = config["swin"]
        self.vision = MaskedSwin(img_size=s["img_size"], patch_size=s["patch_size"],
                                 in_chans=3, embed_dim=s["embed_dim"], depths=s["depths"],
                                 num_heads=s["num_heads"], window_size=s["window_size"],
                                 drop_path_rate=s.get("drop_path_rate", 0.1))
        vision_width = self.vision.num_features                       # 1024
        bert_cfg = BertConfig.from_json_file(config["text_config"])
        bert_cfg.encoder_width = vision_width
        self.text_encoder = BertForMaskedLM.from_pretrained(config["text_encoder"], config=bert_cfg)
        text_width = self.text_encoder.config.hidden_size             # 768

        embed_dim = config["embed_dim"]
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2)
        self.mim_decoder = MIMDecoder(in_dim=vision_width, mask_patch_size=config["mim_mask_patch_size"])
        self.temp = nn.Parameter(torch.ones([]) * config["temp"])
        self.mim_mask_ratio = config["mim_mask_ratio"]
        self.mim_alpha = config["mim_alpha"]

    # --- encoders ---
    def encode_image(self, image, mask=None):
        return self.vision(image, mask=mask)                          # [B,1+49,1024]

    def encode_text(self, text_ids, text_atts):
        return self.text_encoder.bert(text_ids, attention_mask=text_atts,
                                      return_dict=True, mode="text").last_hidden_state

    def cross(self, image_embeds, image_atts, text_embeds, text_atts):
        return self.text_encoder.bert(encoder_embeds=text_embeds, attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,
                                      return_dict=True, mode="fusion").last_hidden_state

    # --- losses ---
    def itc(self, img_cls, txt_cls):
        i = F.normalize(self.vision_proj(img_cls), dim=-1)
        t = F.normalize(self.text_proj(txt_cls), dim=-1)
        logits = i @ t.t() / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

    def itm(self, image_embeds, image_atts, text_embeds, text_atts):
        bs = image_embeds.size(0)
        with torch.no_grad():
            i = F.normalize(self.vision_proj(image_embeds[:, 0]), dim=-1)
            t = F.normalize(self.text_proj(text_embeds[:, 0]), dim=-1)
            w_i2t = F.softmax(i @ t.t() / self.temp, dim=1) + 1e-5
            w_t2i = F.softmax(t @ i.t() / self.temp, dim=1) + 1e-5
            w_i2t.fill_diagonal_(0); w_t2i.fill_diagonal_(0)
        neg_img = torch.stack([image_embeds[torch.multinomial(w_t2i[b], 1).item()] for b in range(bs)])
        neg_txt = torch.stack([text_embeds[torch.multinomial(w_i2t[b], 1).item()] for b in range(bs)])
        neg_txt_atts = torch.stack([text_atts[torch.multinomial(w_i2t[b], 1).item()] for b in range(bs)])
        img_all = torch.cat([image_embeds, image_embeds, neg_img], dim=0)
        txt_all = torch.cat([text_embeds, neg_txt, text_embeds], dim=0)
        atts_all = torch.cat([text_atts, neg_txt_atts, text_atts], dim=0)
        img_atts_all = torch.ones(img_all.size()[:-1], dtype=torch.long, device=image_embeds.device)
        fused = self.cross(img_all, img_atts_all, txt_all, atts_all)[:, 0, :]
        logits = self.itm_head(fused)
        labels = torch.cat([torch.ones(bs, dtype=torch.long),
                            torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(logits.device)
        return F.cross_entropy(logits, labels)

    def mlm(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked, attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True, labels=masked_ids, masked_pos=masked_pos).loss

    def forward(self, image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids):
        # pass 1: FULL image
        image_embeds = self.encode_image(image)                       # [B,1+49,1024]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)
        text_embeds = self.encode_text(text_ids, text_atts)
        loss_itc = self.itc(image_embeds[:, 0], text_embeds[:, 0])
        loss_itm = self.itm(image_embeds, image_atts, text_embeds, text_atts)
        loss_mlm = self.mlm(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)
        # pass 2: MASKED image (shared encoder)
        patch_mask, cell_mask = generate_mim_mask(image.size(0), self.mim_mask_ratio, image.device)
        masked_spatial = self.encode_image(image, mask=patch_mask)[:, 1:, :]   # [B,49,1024]
        loss_mim = mim_loss(self.mim_decoder(masked_spatial), image, cell_mask)
        loss = loss_itc + loss_itm + loss_mlm + self.mim_alpha * loss_mim
        return dict(loss=loss, loss_itc=loss_itc, loss_itm=loss_itm,
                    loss_mlm=loss_mlm, loss_mim=loss_mim)
