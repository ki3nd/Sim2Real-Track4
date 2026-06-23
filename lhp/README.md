# LHP Retriever

## Overview

The **LHP (Local-global Hybrid Perspective)** retriever is a BeiT-3 base/384 image-text retrieval model trained with:
- **LHP crop augmentation**: Stochastic local (random-resized-crop) and global (full-image resize) views
- **Contrastive loss**: ClipLoss over in-batch negatives (no hard sampling per challenge rules)

This module serves as a **stage-1 retriever** in the two-stage retrieval pipeline; retrieved results feed into a later reranking step (out of scope here).

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
| `train_file` | JSON annotation files, **relative to `data_root`** (e.g. `annotation/train/attr_0.json`) |
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
