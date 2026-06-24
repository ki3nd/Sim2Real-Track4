# LHP Masked-View (3rd transform) — Design Spec

**Date:** 2026-06-24
**Scope:** Add a **third training view** to LHP — masked-image — alongside local and global (each 1/3). The masked view randomly masks 40–60% of image patches and **blocks them from attention** (FLIP-style) so the CLS feature only aggregates unmasked patches. Implemented via a **new subclass** of `BEiT3ForRetrieval` (vendored model untouched) that accepts a vision key-padding mask.
**Out of scope:** masking text; eval-time masking; rerank; any change to the original vendored `BEiT3ForRetrieval`.

## 1. Goal & motivation
LHP local/global already lifted R@1 79.221 → 83.873. Add a masked view (grounded in FLIP, "Scaling Language-Image Pre-training via Masking", CVPR 2023): masking forces the image embedding to align with text from partial info → regularization. Three views, **1/3 each**, train-only.

## 2. Mechanism (attention-block via key_padding_mask)
torchscale `BEiT3` routes `encoder_padding_mask` → the encoder as `key_padding_mask`: `attn_weights.masked_fill(key_padding_mask, -inf)` blocks those KEY positions for **all** queries (verified in `multihead_attention`), and zeroes their input embeddings (`x*(1-mask)`). So masked patches are not attended by anyone; unmasked patches attend freely among themselves (no over-masking). CLS (position 0) stays unmasked → CLS aggregates only unmasked patches.

**Constraint:** `BEiT3.forward`'s vision-only branch hardcodes `encoder_padding_mask=None` and offers no vision-mask param. So we bypass `BEiT3.forward` for the masked vision encode and call `vision_embed` + `encoder` directly with our mask — inside a **new subclass**, not by editing anything.

## 3. New class (no edit to vendored BEiT3ForRetrieval)
`lhp/masked_model.py`:
```python
class MaskedBEiT3ForRetrieval(BEiT3ForRetrieval):
    def forward(self, image=None, text_description=None, padding_mask=None,
                vision_padding_mask=None, only_infer=False, **kwargs):
        # vision
        if image is not None and vision_padding_mask is not None:
            x = self.beit3.vision_embed(image)                 # [B, 1+N, D], CLS prepended
            x = self.beit3.encoder(src_tokens=None, encoder_padding_mask=vision_padding_mask,
                                   token_embeddings=x, multiway_split_position=-1)["encoder_out"]
            vision_cls = F.normalize(self.vision_head(x[:, 0, :]), dim=-1)
        elif image is not None:
            x = self.beit3(textual_tokens=None, visual_tokens=image, text_padding_position=None)["encoder_out"]
            vision_cls = F.normalize(self.vision_head(x[:, 0, :]), dim=-1)
        else:
            vision_cls = None
        # text (identical to parent)
        if text_description is not None:
            x = self.beit3(textual_tokens=text_description, visual_tokens=None,
                           text_padding_position=padding_mask)["encoder_out"]
            language_cls = F.normalize(self.language_head(x[:, 0, :]), dim=-1)
        else:
            language_cls = None
        if only_infer:
            return vision_cls, language_cls
        loss, lpi, lpt = self.criterion(vision_cls, language_cls, self.logit_scale.exp())
        return loss, vision_cls, language_cls
```
- `vision_padding_mask=None` → identical to parent `BEiT3ForRetrieval` (eval, baseline unaffected).
- The exact head/normalize/criterion logic mirrors the parent so behavior matches when mask is None.

## 4. Components / files
- **Create `lhp/masked_model.py`** — `MaskedBEiT3ForRetrieval` (above).
- **Modify `lhp/beit3_loader.py`** — `build_beit3_retrieval` builds `MaskedBEiT3ForRetrieval(_get_base_config(img_size=384, drop_path_rate=...))` (strict superset; loads the same state_dict; mask-less forward == old behavior, so `lhp.eval` / baseline are unaffected).
- **Modify `lhp/transform.py`** — add `masked_prob`; 3-way pick local / masked / global. Return `(image_tensor, view, patch_mask)`; `patch_mask` = bool `[num_patches]` (`num_patches=(resolution/16)**2=576` at 384), `True`=masked, `~U(0.4,0.6)` fraction; `None` for non-masked views. Tensor stays at index 0.
- **Modify `lhp/model.py`** — `LHPRetriever.forward(image, text_ids, padding_mask, vision_padding_mask=None)` passes `vision_padding_mask` through.
- **Modify `lhp/dataset.py`** — return `patch_mask` (None when not masked) as a 4th item.
- **Modify `lhp/train.py`** — assemble per-batch `vision_padding_mask` `[B, 1+576]` (CLS col = False; non-masked rows all-False = no-op), pass to `model(...)`.
- **Modify `lhp/config.yaml`** — add `masked_prob` (and keep `local_prob`); train uses local 1/3 + masked 1/3 (+ global remainder). Eval keeps `local_prob=0, masked_prob=0` → global-only.

## 5. Data flow (train)
```
transform → (img, view, patch_mask)         # masked view: patch_mask True for ~U(0.4,0.6)*576 patches
collate   → images[B], patch_masks (list, some None)
train     → vision_padding_mask[B, 577]: row r = [False(CLS)] + patch_mask_r (or all-False if None)
          → loss = model(img, text_ids, padding_mask, vision_padding_mask)   # ClipLoss
```

## 6. Constraints / edges
- CLS never masked (col 0 always False).
- Masked view **train-only**; eval/baseline use `vision_padding_mask=None` → parent path, byte-identical behavior.
- No text masking; no hard sampling; augmentation only → challenge-compliant.
- `transform.__call__` now returns a 3-tuple → update `dataset.py` and `lhp/eval.py` (`_ImageDataset` keeps using `[0]`).
- timm 0.4.12 / transformers 4.x env unchanged.

## 7. Probabilities
`random.random()` thresholds: `< local_prob` → local; `< local_prob+masked_prob` → masked; else global. Defaults (train): `local_prob=1/3`, `masked_prob=1/3` → global 1/3. Eval: both 0.

## 8. Testing
- **transform**: output `(tensor[3,384,384], view, patch_mask)`; masked view → `patch_mask` bool len 576 with fraction in [0.4,0.6]; local/global → `patch_mask is None`; `masked_prob=0` never yields "masked".
- **masked_model (unit, no weights)**: build `MaskedBEiT3ForRetrieval` on CPU; `forward(image, text_ids, padding_mask, vision_padding_mask=None)` returns 3-tuple `(loss, v, l)` (parity with parent); with a `vision_padding_mask` it still returns a scalar loss and runs `loss.backward()`.
- **batch mask assembly (unit)**: a helper builds `[B, 577]` with CLS=False and None→all-False; shape + CLS-col asserts.
- **smoke** (weights+GPU): short DDP train with 3-way transform; loss decreases; checkpoint saved.
