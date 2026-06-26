# UIT DDP Training — Design Spec

**Date:** 2026-06-26
**Scope:** A `uit/` package that trains the merged `models.uit.UIT` model (ITC+ITM+MLM+MIM) on PAB with **torch.distributed (DDP)** — reusing CMP's data pipeline, initializing from X-VLM, and adding cross-rank all-gather to the ITC loss. Produces UIT checkpoints.
**Out of scope:** UIT eval/rerank wiring (already exists via `lhp.rerank_eval` using a CMP reranker; a UIT reranker variant is a later concern); the UIT model itself (already merged).

## 1. Goal
Make UIT trainable under DDP (the model alone is not enough — there is no train loop). Mirror CMP's training harness, drop pose/IHNM, init from X-VLM, add DDP-correct contrastive (all-gather). Challenge-compliant (no hard sampling).

## 2. Files
- **Create `uit/__init__.py`**, **`uit/train.py`** (DDP entry), **`uit/config.yaml`**, **`uit/xvlm_init.py`** (X-VLM checkpoint → UIT key remap loader).
- **Modify `models/uit.py`**: add differentiable cross-rank all-gather to `UIT.itc` (gated on `world_size > 1`).
- Reuse (import): `dataset.search_dataset.search_train_dataset` + `TextMaskingGenerator`, `dataset.create_loader`/`create_sampler`, `transformers.BertTokenizer`, `models.uit.UIT`.

## 3. Model edit — ITC all-gather (DDP correctness)
`UIT.itc` currently computes in-batch cross-entropy on the local batch only. Add a **differentiable** all-gather (GatherLayer/AllGather pattern, like CMP `models/cmp.py` `AllGather` or BeiT-3 `GatherLayer`): when `torch.distributed` is initialized and `world_size > 1`, gather normalized image/text features across ranks so negatives = the global batch; labels offset by `rank * local_bs`. When not distributed (`world_size == 1`), behavior is **identical to the current in-batch path** (so the existing `tests/` stay green and single-process is unchanged). The gather is internal to `UIT.itc`; `UIT.forward` is unchanged.

## 4. X-VLM init (`uit/xvlm_init.py`)
- Load `checkpoint/16m_base_model_state_step_199999.th`.
- **Remap keys** to UIT module names: X-VLM `vision_encoder.*` → `vision.*` (UIT's MaskedSwin); X-VLM text/cross-encoder → `text_encoder.*`; `vision_proj`/`text_proj`/`itm_head` mapped if present. Load with `strict=False`.
- `mask_token` and `mim_decoder` are **not** in X-VLM → keep their random init. Log missing/unexpected keys so the implementer can confirm the vision/text/cross weights actually loaded (not silently skipped due to a name mismatch).
- Positional-embedding interpolation if the X-VLM Swin resolution differs from 224 (reuse the interpolation helper pattern if needed).

## 5. Data pipeline (reuse CMP, pose/hard OFF)
- `search_train_dataset(config, transform)` with `be_pose_img=False`, `be_hard=False` → yields `(image, caption, caption_eda, idx, {}, {}, {}, {})`. Use `image` + `caption` (ignore `caption_eda`, pose, hard — EDA not used by UIT per paper).
- Image transform: train transform at **224** (resize/crop + ImageNet-CLIP normalize, as `dataset/__init__.py` builds from config `h=w=224`).
- `BertTokenizer.from_pretrained(config["text_encoder"])`; `TextMaskingGenerator(tokenizer, mask_prob, max_masks, ...)` + the `mlm(...)` helper (mirror root `train.py`) to build `text_ids_masked`, `masked_pos`, `masked_ids` for the MLM loss.
- `DistributedSampler` + `set_epoch`; `drop_last=True`.

## 6. Train loop (`uit/train.py`, mirrors root `train.py`/`Search.py`)
```
local_rank, rank, world = setup_ddp()                 # init BEFORE building model
model = UIT(config); xvlm_init(model, config["xvlm_ckpt"]); model.to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
optimizer = AdamW(...); scheduler = cosine/step
for epoch:
    sampler.set_epoch(epoch); model.train()
    for (image, caption, *_ ) in loader:
        text = bert_tok(caption, padding=max, max_length, return_tensors)
        text_ids_masked, masked_pos, masked_ids = mlm(caption, text, tokenizer, device, mask_gen, config)
        with autocast(bf16):
            out = model(image, text.input_ids, text.attention_mask,
                        text_ids_masked, masked_pos, masked_ids)   # dict of 5 losses
        out["loss"].backward(); optimizer.step(); scheduler.step(); zero_grad()
        if rank==0 and i%50==0: print epoch/step/losses
    if rank==0: save {"model": model.module.state_dict(), "config"} -> output_dir/uit_epoch{e}.pth
```
- ITC all-gather happens inside `UIT.itc` (Section 3). MIM mask generated inside `UIT.forward`.
- `find_unused_parameters=True` (BERT may have unused params; safe default, matching CMP/LHP).

## 7. Config (`uit/config.yaml`)
```yaml
data_root: 'data/PAB'
train_file: [ 'annotation/train/imgs_0.json', ... ]   # relative to data_root (like lhp)
image_res: 224
swin: { img_size: 224, patch_size: 4, embed_dim: 128, depths: [2,2,18,2], num_heads: [4,8,16,32], window_size: 7, drop_path_rate: 0.1 }
text_config: 'configs/config_bert.json'
text_encoder: 'checkpoint/bert-base-uncased'
xvlm_ckpt: 'checkpoint/16m_base_model_state_step_199999.th'
embed_dim: 256
temp: 0.07
max_tokens: 56
mask_prob: 0.25
max_masks: 10
mim_mask_patch_size: 32
mim_mask_ratio: 0.6
mim_alpha: 0.1356
batch_size: 84            # paper; reduce per-GPU + grad-accum if OOM
lr: 1.0e-5
epochs: 22
weight_decay: 0.01
output_dir: 'output/uit'
```

## 8. Prerequisites (user-managed)
X-VLM `checkpoint/16m_base_model_state_step_199999.th`; `checkpoint/bert-base-uncased`; `configs/config_bert.json`; swin 22k ckpt if `load_params` used; PAB train data + `sentencepiece` not needed (UIT uses BERT, not spm).

## 9. Constraints
- DDP init BEFORE model build (so `UIT.itc` reads correct `world_size`/`rank` for gather).
- No pose, no IHNM (challenge-compliant). EDA unused.
- `UIT.itc` all-gather must be **differentiable** (gradients flow to local features) and a **no-op when `world_size==1`** (existing tests unchanged).
- `uit/` imports `models.uit` + CMP data helpers (`dataset.*`) — NOT `models.cmp`/`Search.py` model classes.

## 10. Testing
- **ITC gather no-op parity (no weights, single-process):** with `world_size==1` (no dist init), the modified `UIT.itc` returns the same value as the pre-edit in-batch CE on the same inputs (unit test on the `itc` method with random feature tensors; existing `tests/` still pass).
- **xvlm_init key remap (unit, no full model):** feed a tiny fake state_dict with `vision_encoder.*`/text keys into the remap function; assert keys are renamed to `vision.*`/`text_encoder.*` and `mask_token`/`mim_decoder` are reported as not-loaded (missing).
- **train.py parse + config load:** `ast.parse`; config loads with expected keys.
- **Smoke (DEFERRED — needs X-VLM + bert + GPU + PAB):** short 1-epoch DDP run on a tiny train_file; loss prints and decreases; checkpoint saved; confirm the all-gather path runs across 2 ranks (loss differs from single-process). Document when prereqs exist.
