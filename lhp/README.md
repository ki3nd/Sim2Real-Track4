# LHP Retriever

## Overview

The **LHP (Local-global Hybrid Perspective)** retriever is a BeiT-3 base/384 image-text retrieval model trained with:
- **LHP crop augmentation**: Stochastic local (random-resized-crop) and global (full-image resize) views
- **Contrastive loss**: ClipLoss over in-batch negatives (no hard sampling per challenge rules)

This module serves as a **stage-1 retriever** in the two-stage retrieval pipeline; retrieved results can be reranked by CMP's pose-aware cross-encoder via `lhp.rerank_eval`.

## Prerequisites

### Install Dependencies

```bash
# Install sentencepiece (required by XLMRobertaTokenizer)
pip install sentencepiece
# OR via uv
uv pip install sentencepiece
```

`torchscale` and `timm>=0.4.12` are already required by the project.

### Download Checkpoints

Download the pretrained BeiT-3 model and tokenizer:

```bash
wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -P checkpoint/
wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -P checkpoint/
```

Verify that `checkpoint/` contains:
- `beit3.spm`
- `beit3_base_patch16_384_coco_retrieval.pth`

## Training

Launch distributed training via DDP on multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 -m lhp.train --config lhp/config.yaml
```

Replace `--nproc_per_node=2` with your number of GPUs and `CUDA_VISIBLE_DEVICES` with your GPU indices.

**Key points:**
- DDP is initialized **before** model construction (required for ClipLoss to gather negatives correctly across ranks)
- `sampler.set_epoch(epoch)` ensures different data shuffling each epoch
- Default batch size is 184 (configured in `lhp/config.yaml`); see [VRAM Guidance](#vram-guidance) if you encounter OOM

## Evaluation (PAB test)

Score the dual-encoder retriever on the PAB test set (R@1/R@5/R@10, mAP, mINP), reusing CMP's `eval.mAP`. Single GPU, no DDP, no cross-encoder rerank — stage-1 cosine retrieval only. Run once per checkpoint and compare.

```bash
# baseline: zero-shot BeiT-3 COCO checkpoint (not fine-tuned on PAB)
.venv/bin/python -m lhp.eval --config lhp/config.yaml --kind beit3 \
    --checkpoint checkpoint/beit3_base_patch16_384_coco_retrieval.pth

# after LHP fine-tuning
.venv/bin/python -m lhp.eval --config lhp/config.yaml --kind lhp \
    --checkpoint output/lhp/lhp_epoch2.pth
```

The R@1 / mAP delta between the two runs is the LHP improvement.

- `--kind beit3`: load a raw `BEiT3ForRetrieval` checkpoint (the COCO `.pth`).
- `--kind lhp`: load a trained `lhp_epoch*.pth` (`{"model": ...}` wrapper checkpoint).
- Test set: `test_file` in `lhp/config.yaml` (relative to `data_root`), CMP format — each record `{image, image_id, caption: [list]}`; match query↔gallery by `image_id`. Test images resolve as `data_root + ann["image"]`.
- Eval uses a deterministic global-only view (no LHP local crop).

## Two-Stage Rerank Evaluation

Run LHP as the stage-1 retriever, then rerank each query's top-k candidates with the trained CMP pose-aware cross-encoder. This reports the final R@1/R@5/R@10, mAP, and mINP using CMP's existing `evaluation_itc`, `evaluation_itm`, and `mAP` code.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m lhp.rerank_eval \
    --lhp-config lhp/config.yaml \
    --lhp-checkpoint output/lhp/lhp_epoch2.pth \
    --lhp-kind lhp \
    --cmp-config configs/cmp.yaml \
    --cmp-checkpoint path/to/cmp_pose_nohard_checkpoint.pth
```

For a zero-shot BeiT-3 stage-1 baseline, switch the LHP checkpoint and kind:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m lhp.rerank_eval \
    --lhp-config lhp/config.yaml \
    --lhp-checkpoint checkpoint/beit3_base_patch16_384_coco_retrieval.pth \
    --lhp-kind beit3 \
    --cmp-config configs/cmp.yaml \
    --cmp-checkpoint path/to/cmp_pose_nohard_checkpoint.pth
```

Prerequisites:
- Single GPU with both CMP and LHP dependencies installed.
- `timm==0.4.12` is supported; CMP's Swin import falls back to `timm.models.layers` for this BeiT-3-compatible version.
- `transformers` 4.x with `sentencepiece` installed for `XLMRobertaTokenizer("checkpoint/beit3.spm")`.
- A trained CMP checkpoint matching `configs/cmp.yaml`; it should be pose-aware and no-hard for this rerank setup.
- PAB test images and pose images must be available under CMP's `image_root`; pose paths are resolved as `pose/<ann["image"]>`.

Important alignment detail: `lhp.rerank_eval` drives both stages from the same CMP `search_test_dataset`, so gallery/query ordering is shared by index. The LHP similarity matrix is guarded to be `[N_query, N_gallery]` before CMP reranking.

## Configuration

Edit `lhp/config.yaml` to adjust:

| Parameter | Purpose |
|-----------|---------|
| `resolution` | Image size (default: 384) |
| `crop_scale` | RandomResizedCrop scale range for local view (default: [0.5, 0.8]) |
| `local_prob` | Probability of local view (default: 0.5) |
| `batch_size` | Effective global batch size (default: 184) |
| `epochs` | Number of training epochs (default: 3) |
| `max_words` | Max text length before tokenization (default: 56) |
| `max_tokens` | Max token count post-tokenization (default: 64) |
| `lr` | Learning rate (default: 1e-5) |
| `weight_decay` | AdamW weight decay (default: 0.01) |
| `drop_path_rate` | DropPath rate in BeiT-3 (default: 0.1) |
| `eda` | Enable EDA augmentation (default: false) |
| `data_root` | Dataset root — prefixes both annotation files and image paths (move the dataset by changing only this) |
| `train_file` | JSON annotation files, **relative to `data_root`** (e.g. `annotation/train/imgs_0.json`) |
| `test_file` | Test annotation, relative to `data_root` (e.g. `annotation/test/attr.json`) — used by `lhp.eval` |
| `batch_size_eval` | Batch size for evaluation encoding (default: 64) |
| `beit3_ckpt` | Path to BeiT-3 checkpoint |
| `spm_model` | Path to BeiT-3 tokenizer model |
| `output_dir` | Checkpoint output directory |

## Inference

### Encode Images

```python
from lhp.model import LHPRetriever
import torch

model = LHPRetriever(ckpt_path="checkpoint/beit3_base_patch16_384_coco_retrieval.pth")
model = model.eval()

# Prepare image (3, 384, 384) tensor
image = torch.randn(1, 3, 384, 384)
with torch.no_grad():
    img_embed = model.encode_image(image)  # (1, hidden_dim)
```

### Encode Text

```python
from transformers import XLMRobertaTokenizer
from lhp.tokenization import tokenize_caption
import torch

tokenizer = XLMRobertaTokenizer("checkpoint/beit3.spm")
text_ids, padding_mask = tokenize_caption(tokenizer, "person walking", max_len=64)

with torch.no_grad():
    txt_embed = model.encode_text(
        torch.tensor([text_ids]),
        torch.tensor([padding_mask])
    )  # (1, hidden_dim)
```

### Compute Similarity

```python
from lhp.infer import similarity

sim = similarity(img_embed, txt_embed)  # [N_txt, N_img] — text-to-image orientation
```

## VRAM Guidance

Batch size 184 with BeiT-3 base/384 may OOM on 4×RTX 3090 (24G per GPU).

**If OOM occurs:**
1. Reduce `batch_size` in `config.yaml` (e.g., 92 → per-GPU 46)
2. Add gradient accumulation:
   ```python
   # In train.py, wrap backward in accumulation
   loss.backward()
   if (i + 1) % accum_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```
3. Keep the **effective global batch size** near 184 so contrastive negatives scale appropriately

The contrastive loss negatives scale with the number of samples across all GPUs in each step, so effective batch size directly impacts training dynamics.

## Constraints

- **No hard sampling**: Only in-batch negatives via ClipLoss; no external hard mining
- **LHP augmentation**: Stochastic choice between local (random-resized-crop) and global (full resize) views per image
- **Contrastive loss**: As per the paper; no index-based grouping or additional hard-negative pairing logic

## Testing

Run unit tests for transform, tokenization, and dataset:

```bash
.venv/bin/python -m pytest lhp/tests/ -v
```

Full training and inference smoke tests require the prerequisites (sentencepiece, checkpoints) above.

## Output

Training produces checkpoints in `output/lhp/` (configured in `config.yaml`):

```
output/lhp/
├── lhp_epoch0.pth  # Contains model state dict + config
├── lhp_epoch1.pth
└── lhp_epoch2.pth
```

Each checkpoint is a dictionary: `{"model": state_dict, "config": config_dict}`.

## References

- BeiT-3: https://github.com/microsoft/unilm/tree/master/beit3
- timm: https://timm.fast.ai/
- XLM-R Tokenizer: https://huggingface.co/transformers/model_doc/xlmroberta.html
