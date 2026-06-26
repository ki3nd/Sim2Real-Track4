# UIT Model (Unified Image-Text, with MIM) — Design Spec

**Date:** 2026-06-26
**Scope:** A standalone **UIT** model (Swin-B + BERT cross-encoder) trained with four objectives — **ITC + ITM + MLM + MIM** — for text-based person anomaly retrieval. Focus: the **MIM (Masked Image Modeling)** branch (SimMIM-style: a `MaskedSwin` that injects a learnable mask token, plus a lightweight linear decoder). Coded as **separate files** that reuse `models/` building blocks; **no integration with `cmp.py`**, **no pose**, **no IHNM hard-negative mining**.
**Out of scope:** training loop/eval wiring (separate later sub-projects), distractor gallery, rerank.

## 1. Goal
Reproduce the Hybrid paper's UIT (Sec 2.2): a unified image-text model adding masked image reconstruction (MIM) on top of the CMP-style contrastive + matching + MLM stack. UIT later serves (with LHP feature-selection) as the reranker. This spec covers the **model only** (the `nn.Module` + its forward computing the 4 losses).

## 2. Architecture (standalone, reuses building blocks)
- **Vision encoder:** `MaskedSwin(SwinTransformer)` — subclass of `models/swin_transformer.py` (NOT modified). Override `forward(x, mask=None)`: `mask=None` ⇒ identical to parent; `mask` given ⇒ replace masked patch tokens with a learnable `mask_token` right after `patch_embed`.
- **Text + cross encoder:** BERT via `from models.bert import BertConfig, BertForMaskedLM` (mode `'text'` for text encode, `'fusion'` for cross-attention) — same as CMP uses, but instantiated inside UIT.
- **Heads (UIT's own):** `vision_proj = Linear(1024, embed_dim)` / `text_proj = Linear(768, embed_dim)` applied to the **CLS token** then L2-normalized (ITC); `itm_head` (2-class MLP on fused CLS); MLM head (in `BertForMaskedLM`); `mim_decoder` (linear, SimMIM).
- **No** pose branch (`models/pose.py`), **no** IHNM (`get_matching_loss_hard`). Image size **224** (Swin-B patch4 window7 224; total stride 32 → 7×7=49 final tokens, 1024-d).

## 3. Files (separate, under `models/`, reusing swin/bert)
- **Create `models/masked_swin.py`** — `MaskedSwin` + `mask_token` + SimMIM `generate_mim_mask(...)` helper.
- **Create `models/uit.py`** — `class UIT(nn.Module)`: builds MaskedSwin + BERT + heads + `mim_decoder`; methods for the 4 losses; `forward` returning them. No import of `models.cmp`.
- Reuse (import, not modify): `models/swin_transformer.py` (`SwinTransformer`, `interpolate_relative_pos_embed`), `models/bert.py`.

## 4. The 4 losses + forward (2 vision passes)
```
# pass 1: FULL image → ITC / ITM / MLM
img_embeds  = masked_swin(image)             # [B, 1+49, 1024]  (mask=None)
text_embeds = bert(text, mode='text')        # [B, T, 768]
L_itc = contrastive(vision_proj(img_embeds[:,0]), text_proj(text_embeds[:,0]))   # CLS-pool, in-batch
L_itm = matching(img_embeds, text_embeds)    # cross-encoder fusion + itm_head, in-batch negatives
L_mlm = mlm(text_masked, img_embeds)         # cross-encoder + mlm head

# pass 2: MASKED image → MIM
masked_spatial = masked_swin(image, mask=mim_mask)[:, 1:, :]   # [B, 49, 1024] (drop CLS)
recon = mim_decoder(masked_spatial)                            # → [B, 3, 224, 224]
L_mim = l1(recon, image) over masked patches only

L = L_itc + L_itm + L_mlm + alpha * L_mim     # alpha = 0.1356 (paper)
```
Per the paper (Fig 1b): the FULL image feeds ITC/ITM/MLM; a separately-masked image feeds MIM. Two Swin forward passes per step (~2× vision compute) — accepted.

## 5. MIM detail (SimMIM recipe) — the focus
- **mask_patch_size = 32** (= Swin total stride) → mask grid 224/32 = **7×7 = 49** cells. **mask ratio = `mim_mask_ratio` (config, default 0.6 = SimMIM default; the Hybrid paper cites SimMIM but does not state a ratio).**
- **`generate_mim_mask`**: random-sample `round(0.6*49)` of the 49 cells as masked → bool. Expand to the **stage-0 patch grid** (56×56; each 32-cell = 8×8 patches) for token replacement, and keep the 7×7 (or 224×224) form for the loss region.
- **`MaskedSwin.forward(x, mask)`**: `patch_embed(x)` → `[B,3136,128]`; where `mask` True, overwrite that token with `mask_token` (`nn.Parameter[1,1,embed_dim=128]`); then the unchanged layers/norm; return `[B, 1+49, 1024]`.
- **`mim_decoder`**: `nn.Linear(1024, 32*32*3 = 3072)` on the 49 final tokens → `[B,49,3072]` → reshape/PixelShuffle to `[B,3,224,224]`.
- **`get_mim_loss`**: `F.l1_loss(recon, image, reduction='none')`, averaged over **masked pixels only** (mask broadcast to pixel resolution). (SimMIM eq. 4–5.)
- `mask_token` lives at the patch_embed output dim (**128** for Swin-B stage-0), not 1024.

## 6. Constraints
- `MaskedSwin(mask=None)` ⇒ byte-identical to `SwinTransformer` (so the subclass is a safe drop-in; `swin_transformer.py` untouched).
- ITC/ITM/MLM see the FULL (unmasked) image; only MIM uses the masked image.
- No hard sampling / no pose → challenge-compliant.
- Standalone: `models/uit.py` and `models/masked_swin.py` must NOT import `models/cmp.py` or `models/model_search.py`.
- Initialization from X-VLM (as CMP does) is a training-time concern (out of scope here); the model must build from config with random/ImageNet-Swin init and load a checkpoint later.

## 7. Config (UIT model knobs)
```yaml
image_res: 224
mim_mask_patch_size: 32
mim_mask_ratio: 0.6           # SimMIM default (paper cites SimMIM, no ratio given)
mim_alpha: 0.1356            # paper
embed_dim: 256               # ITC projection dim (CLS-pooled vision/text → Linear → 256, L2-norm)
temp: 0.07
```

## 8. Testing (model-level, no training)
- **MaskedSwin parity (no weights, CPU):** `masked_swin(x)` == `SwinTransformer(x)` for the same weights when `mask=None` (allclose); with a mask, output differs and shape is unchanged `[B,1+49,1024]`.
- **generate_mim_mask:** returns bool with the right grid size; masked fraction ≈ `mim_mask_ratio`; expansion to 56×56 / pixel grid has correct shapes.
- **mim_decoder + loss:** decoder maps `[B,49,1024] → [B,3,224,224]`; `get_mim_loss` is a scalar, `backward()` runs; loss counts masked region only (a fully-unmasked mask → zero contribution).
- **UIT.forward smoke (CPU, tiny batch):** returns the 4 loss terms (or their sum) as scalars; `backward()` runs. (Distributed ITC gather is a train-time concern; test the single-process path.)
