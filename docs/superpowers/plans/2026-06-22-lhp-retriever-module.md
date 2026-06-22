# LHP Retriever Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `lhp/` package that trains a BeiT-3 base/384 retriever with Local-global Hybrid Perspective (LHP) crop augmentation, optimized by contrastive loss, producing a retriever checkpoint.

**Architecture:** Vendored BeiT-3 (trimmed) under `lhp/beit3/`, wrapped by `LHPRetriever`. A stochastic local/global transform feeds a DDP contrastive training loop using BeiT-3's built-in `ClipLoss` (differentiable cross-rank gather). Train-only scope; an `infer.py` exposes encode/similarity for a later rerank step. No coupling to CMP `Search.py`.

**Tech Stack:** PyTorch (DDP), torchscale, timm 0.4.12, transformers `XLMRobertaTokenizer` + sentencepiece, BeiT-3 (`beit3_base_patch16_384_retrieval`).

## Global Constraints

- **No hard sampling** (challenge rule) — contrastive uses in-batch negatives only (ClipLoss). Do not add IHNM/hard pairs.
- **Reproducibility** — vendor BeiT-3 into `lhp/beit3/`; never import from `/home/pc1175/Code/open-sources`.
- **No coupling** — `lhp/` must not import `Search.py`, `models/cmp.py`, `models/model_search.py`.
- **Contrastive per paper** — ClipLoss diagonal positives; no `idx`-grouping.
- **BeiT-3 variant** — `beit3_base_patch16_384_retrieval`, init from `checkpoint/beit3_base_patch16_384_coco_retrieval.pth`, tokenizer `checkpoint/beit3.spm`.
- **Image normalize** — `IMAGENET_INCEPTION_MEAN/STD` = (0.5,0.5,0.5)/(0.5,0.5,0.5); resolution 384.
- **Tokens** — `max_tokens = 64`; sequence = `[bos] + ids + [eos]` then pad to 64; `padding_mask`: 0 = real, 1 = pad.
- **DDP** — initialize distributed BEFORE building the model (so `ClipLoss(rank, world_size)` is correct); use `DistributedSampler` + `sampler.set_epoch(epoch)`.

**Prerequisites (USER-MANAGED — verify before Task 5):**
- `.venv` has `torchscale`, `sentencepiece`, `timm==0.4.12`, `transformers`. (Probe showed `sentencepiece` missing — install it.)
- `checkpoint/beit3.spm` and `checkpoint/beit3_base_patch16_384_coco_retrieval.pth` downloaded.
- `pytest` available: `.venv/bin/python -m pytest --version` (else `uv add --dev pytest`).

---

## File Structure

```
lhp/
├── __init__.py
├── beit3/                  # vendored, trimmed BeiT-3
│   ├── utils.py            # TRIMMED (dist helpers + ClipLoss + load helpers only)
│   ├── modeling_utils.py   # verbatim copy
│   └── modeling_finetune.py# verbatim copy
├── beit3_loader.py         # puts lhp/beit3 on sys.path, re-exports symbols
├── transform.py            # LHPTransform
├── tokenization.py         # tokenize_caption()
├── dataset.py              # LHPDataset
├── model.py                # LHPRetriever
├── train.py                # DDP training entry
├── infer.py                # encode/similarity API
├── config.yaml
└── tests/
    ├── test_transform.py
    ├── test_tokenization.py
    └── test_dataset.py
```

Reused from CMP (import, not copy): `dataset/utils.py` (`pre_caption`, `read_json_to_list`), `dataset/eda.py` (only if `eda: true`).

---

## Task 1: Scaffold package + vendor trimmed BeiT-3

**Files:**
- Create: `lhp/__init__.py` (empty)
- Create: `lhp/beit3/modeling_utils.py` (verbatim copy)
- Create: `lhp/beit3/modeling_finetune.py` (verbatim copy)
- Create: `lhp/beit3/utils.py` (trimmed copy)
- Create: `lhp/beit3_loader.py`
- Test: `lhp/tests/test_import_model.py`

**Interfaces:**
- Produces: `lhp.beit3_loader.build_beit3_retrieval(drop_path_rate=0.1) -> nn.Module`, `lhp.beit3_loader.load_pretrained(model, ckpt_path)`, and re-exports `XLMRobertaTokenizer` usage helpers via beit3 path.

- [ ] **Step 1: Copy the two verbatim BeiT-3 files + make `lhp/beit3` a package**

```bash
mkdir -p lhp/beit3 lhp/tests
touch lhp/__init__.py lhp/tests/__init__.py lhp/beit3/__init__.py
cp /home/pc1175/Code/open-sources/unilm/beit3/modeling_utils.py    lhp/beit3/modeling_utils.py
cp /home/pc1175/Code/open-sources/unilm/beit3/modeling_finetune.py lhp/beit3/modeling_finetune.py
```

`lhp/beit3/` is a real Python package (note the `__init__.py`). This avoids
`sys.path` injection and the resulting `import utils` name collision with the
CMP repo-root `utils.py` (whichever loaded first would otherwise win
`sys.modules['utils']` and break the other component).

- [ ] **Step 2: Create the trimmed `lhp/beit3/utils.py`**

Copy ONLY these symbols **verbatim** from `/home/pc1175/Code/open-sources/unilm/beit3/utils.py`, in this order, into a new file whose ONLY module-level imports are the four lines shown. Do NOT copy `from torch._six import inf`, `tensorboardX`, `torchmetrics`, or `timm.utils`.

Header of the new file:
```python
# Trimmed from BeiT-3 utils.py (MIT, Microsoft) — only symbols needed by the retrieval model.
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
```
Then copy these functions/classes verbatim (exact upstream line ranges):
- `is_dist_avail_and_initialized` (238–244)
- `get_world_size` (246–250)
- `get_rank` (252–255)
- `load_state_dict` (326–end of function)
- `load_model_and_may_interpolate` (521–583)
- `GatherLayer` (652–666)
- `gather_features` (669–678)
- `ClipLoss` (682–728)

(These functions reference only `torch`, `dist`, `nn`, `F` — verified.)

- [ ] **Step 3: Patch the vendored `modeling_finetune.py` to package-relative imports**

The upstream file uses flat imports that assume its dir is on `sys.path`.
Change exactly these two lines so they resolve within the `lhp.beit3` package
(this is the ONLY edit to the vendored files):

```python
# BEFORE (upstream)
import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config

# AFTER (package-relative)
from . import utils
from .modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
```

`modeling_utils.py` needs no change (it imports only `torchscale` and `timm`).

- [ ] **Step 4: Create `lhp/beit3_loader.py`**

```python
from lhp.beit3.modeling_finetune import beit3_base_patch16_384_retrieval
from lhp.beit3 import utils as beit3_utils


def build_beit3_retrieval(drop_path_rate: float = 0.1):
    """Build BEiT3ForRetrieval (base, patch16, 384). Distributed must be initialized
    BEFORE calling this so the internal ClipLoss gets correct rank/world_size."""
    return beit3_base_patch16_384_retrieval(pretrained=False, drop_path_rate=drop_path_rate)


def load_pretrained(model, ckpt_path: str):
    """Load a BeiT-3 retrieval checkpoint with positional-embedding interpolation."""
    beit3_utils.load_model_and_may_interpolate(
        ckpt_path, model, model_key="model|module", model_prefix="")
    return model
```

- [ ] **Step 5: Write the failing test**

```python
# lhp/tests/test_import_model.py
import torch.nn as nn
from lhp.beit3_loader import build_beit3_retrieval, beit3_utils


def test_build_beit3_retrieval_returns_module_with_cliploss():
    model = build_beit3_retrieval(drop_path_rate=0.0)
    assert isinstance(model, nn.Module)
    # ClipLoss is the contrastive criterion baked into BEiT3ForRetrieval
    assert type(model.criterion).__name__ == "ClipLoss"


def test_trimmed_utils_has_no_broken_imports():
    # importing beit3_utils must succeed despite upstream torch._six/tensorboardX
    assert hasattr(beit3_utils, "load_model_and_may_interpolate")
    assert hasattr(beit3_utils, "ClipLoss")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_import_model.py -v`
Expected: FAIL/ERROR initially if any vendored import is wrong (e.g. ModuleNotFoundError). Fix the trimmed `utils.py` / relative-import patch until imports resolve.

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest lhp/tests/test_import_model.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add lhp/__init__.py lhp/beit3/ lhp/beit3_loader.py lhp/tests/
git commit -m "feat(lhp): scaffold package and vendor trimmed BeiT-3 retrieval model"
```

---

## Task 2: LHP transform (local/global view)

**Files:**
- Create: `lhp/transform.py`
- Test: `lhp/tests/test_transform.py`

**Interfaces:**
- Produces: `LHPTransform(resolution=384, crop_scale=(0.5,0.8), local_prob=0.5)`; `__call__(pil_image) -> (tensor[3,res,res], view_str)` where `view_str in {"local","global"}`.

- [ ] **Step 1: Write the failing test**

```python
# lhp/tests/test_transform.py
from PIL import Image
import torch
from lhp.transform import LHPTransform


def _img():
    return Image.new("RGB", (640, 480), (123, 117, 104))


def test_output_shape_is_resolution():
    t = LHPTransform(resolution=384)
    out, view = t(_img())
    assert isinstance(out, torch.Tensor) and out.shape == (3, 384, 384)
    assert view in ("local", "global")


def test_local_prob_zero_is_always_global():
    t = LHPTransform(resolution=384, local_prob=0.0)
    assert all(t(_img())[1] == "global" for _ in range(20))


def test_local_prob_one_is_always_local():
    t = LHPTransform(resolution=384, local_prob=1.0)
    assert all(t(_img())[1] == "local" for _ in range(20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_transform.py -v`
Expected: FAIL with `ModuleNotFoundError: lhp.transform`.

- [ ] **Step 3: Implement `lhp/transform.py`**

```python
import random
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation


class LHPTransform:
    """Stochastically pick a local (random-resized-crop) or global (full resize) view,
    then ToTensor + Inception normalize. Returns (tensor, view_name)."""

    def __init__(self, resolution=384, crop_scale=(0.5, 0.8), local_prob=0.5):
        self.local_prob = local_prob
        self._local = transforms.Compose([
            RandomResizedCropAndInterpolation(resolution, scale=crop_scale, interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
        ])
        self._global = transforms.Resize((resolution, resolution), interpolation=3)  # BICUBIC
        self._finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    def __call__(self, image):
        if random.random() < self.local_prob:
            view, img = "local", self._local(image)
        else:
            view, img = "global", self._global(image)
        return self._finalize(img), view
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest lhp/tests/test_transform.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lhp/transform.py lhp/tests/test_transform.py
git commit -m "feat(lhp): add LHP local/global view transform"
```

---

## Task 3: Caption tokenization helper

**Files:**
- Create: `lhp/tokenization.py`
- Test: `lhp/tests/test_tokenization.py`

**Interfaces:**
- Produces: `tokenize_caption(tokenizer, text: str, max_len: int) -> (ids: list[int], padding_mask: list[int])`. `len(ids) == len(padding_mask) == max_len`; `padding_mask` 0 = real, 1 = pad.

- [ ] **Step 1: Write the failing test** (uses a stub tokenizer — no spm needed)

```python
# lhp/tests/test_tokenization.py
from lhp.tokenization import tokenize_caption


class StubTok:
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    def tokenize(self, text):
        return text.split()
    def convert_tokens_to_ids(self, toks):
        return [10 + i for i, _ in enumerate(toks)]


def test_pads_to_max_len_and_marks_padding():
    ids, mask = tokenize_caption(StubTok(), "a b c", max_len=8)
    assert len(ids) == 8 and len(mask) == 8
    assert ids[0] == 0 and ids[4] == 2          # bos ... eos  (3 words + bos + eos = 5 real)
    assert ids[5:] == [1, 1, 1]                 # pad
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1]


def test_truncates_long_text():
    ids, mask = tokenize_caption(StubTok(), " ".join(["w"] * 100), max_len=8)
    assert len(ids) == 8
    assert ids[0] == 0 and ids[7] == 2          # bos at 0, eos at last, 6 content
    assert mask == [0] * 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_tokenization.py -v`
Expected: FAIL with `ModuleNotFoundError: lhp.tokenization`.

- [ ] **Step 3: Implement `lhp/tokenization.py`**

```python
def tokenize_caption(tokenizer, text, max_len):
    """Tokenize a raw caption the BeiT-3 way: [bos] + ids[:max_len-2] + [eos], pad to max_len.
    padding_mask: 0 for real tokens, 1 for padding."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_len - 2:
        ids = ids[:max_len - 2]
    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    num_tokens = len(ids)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    ids = ids + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return ids, padding_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest lhp/tests/test_tokenization.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add lhp/tokenization.py lhp/tests/test_tokenization.py
git commit -m "feat(lhp): add BeiT-3 caption tokenization helper"
```

---

## Task 4: LHPDataset

**Files:**
- Create: `lhp/dataset.py`
- Test: `lhp/tests/test_dataset.py`

**Interfaces:**
- Consumes: `dataset.utils.read_json_to_list`, `dataset.utils.pre_caption`, `lhp.transform.LHPTransform`.
- Produces: `LHPDataset(ann_files: list[str], image_root: str, transform, max_words=56, eda=False, eda_p=0.5)`; `__getitem__ -> (image_tensor, caption_str, image_path)`; robust to a corrupt image (resamples).

- [ ] **Step 1: Write the failing test** (builds a tiny jsonl + image in tmp)

```python
# lhp/tests/test_dataset.py
import json, os
from PIL import Image
from lhp.transform import LHPTransform
from lhp.dataset import LHPDataset


def _setup(tmp_path):
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    Image.new("RGB", (320, 240), (100, 110, 120)).save(root / "train" / "0.jpg")
    ann = tmp_path / "ann.jsonl"
    with open(ann, "w") as f:
        f.write(json.dumps({"image": "train/0.jpg",
                            "caption": "a person is running on grass"}) + "\n")
    return str(ann), str(root)


def test_returns_image_caption_path(tmp_path):
    ann, root = _setup(tmp_path)
    ds = LHPDataset([ann], root, LHPTransform(resolution=384), eda=False)
    assert len(ds) == 1
    img, cap, path = ds[0]
    assert tuple(img.shape) == (3, 384, 384)
    assert isinstance(cap, str) and len(cap) > 0
    assert path.endswith("train/0.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: lhp.dataset`.

- [ ] **Step 3: Implement `lhp/dataset.py`**

```python
import os
import random
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import pre_caption, read_json_to_list  # reused from CMP (I/O only)


class LHPDataset(Dataset):
    def __init__(self, ann_files, image_root, transform, max_words=56, eda=False, eda_p=0.5):
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.eda = eda
        self.eda_p = eda_p
        self.ann = []
        for f in ann_files:
            self.ann.extend(read_json_to_list(f))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self.ann) - 1))
        image, _view = self.transform(image)
        caption = pre_caption(ann["caption"], self.max_words, self.eda, self.eda_p)
        return image, caption, ann["image"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest lhp/tests/test_dataset.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add lhp/dataset.py lhp/tests/test_dataset.py
git commit -m "feat(lhp): add LHPDataset (no pose/hard, optional eda)"
```

---

## Task 5: LHPRetriever model wrapper + infer API

**Files:**
- Create: `lhp/model.py`
- Create: `lhp/infer.py`

**Interfaces:**
- Consumes: `lhp.beit3_loader.build_beit3_retrieval`, `load_pretrained`.
- Produces:
  - `LHPRetriever(ckpt_path=None, drop_path_rate=0.1)` → `nn.Module` wrapping `BEiT3ForRetrieval`.
  - `forward(image, text_ids, padding_mask) -> (loss, vision_cls, language_cls)`.
  - `encode_image(image) -> vision_cls`, `encode_text(text_ids, padding_mask) -> language_cls`.
  - `lhp.infer.similarity(img_feats, txt_feats) -> Tensor[N_txt, N_img]`.

**Note:** Smoke-verified (needs weights + likely GPU), not a pytest unit test.

- [ ] **Step 1: Implement `lhp/model.py`**

```python
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

    def forward(self, image, text_ids, padding_mask):
        # BEiT3ForRetrieval computes ClipLoss internally over in-batch (gathered) negatives
        return self.beit3(image=image, text_description=text_ids, padding_mask=padding_mask)

    @torch.no_grad()
    def encode_image(self, image):
        vision_cls, _ = self.beit3(image=image, only_infer=True)
        return vision_cls

    @torch.no_grad()
    def encode_text(self, text_ids, padding_mask):
        _, language_cls = self.beit3(text_description=text_ids, padding_mask=padding_mask, only_infer=True)
        return language_cls
```

- [ ] **Step 2: Implement `lhp/infer.py`**

```python
import torch


def similarity(img_feats, txt_feats):
    """Cosine similarity (features are already L2-normalized by the model heads).
    Returns [N_txt, N_img]."""
    return txt_feats @ img_feats.t()


def topk(sim_t2i, k):
    """Top-k image indices per text query. Returns (values, indices), each [N_txt, k]."""
    return sim_t2i.topk(k=min(k, sim_t2i.size(1)), dim=1)
```

- [ ] **Step 3: Smoke test (requires prereqs — weights + spm)**

Create a throwaway script and run on 1 GPU:
```bash
.venv/bin/python - <<'PY'
import torch
from transformers import XLMRobertaTokenizer
from lhp.model import LHPRetriever
from lhp.tokenization import tokenize_caption

tok = XLMRobertaTokenizer("checkpoint/beit3.spm")
m = LHPRetriever(ckpt_path="checkpoint/beit3_base_patch16_384_coco_retrieval.pth").cuda().eval()

img = torch.randn(2, 3, 384, 384).cuda()
ids, mask = zip(*[tokenize_caption(tok, "a person running", 64) for _ in range(2)])
ids = torch.tensor(ids).cuda(); mask = torch.tensor(mask).cuda()

vf = m.encode_image(img); tf = m.encode_text(ids, mask)
from lhp.infer import similarity
print("vision", vf.shape, "text", tf.shape, "sim", similarity(vf, tf).shape)
assert vf.shape[0] == 2 and tf.shape[0] == 2
print("SMOKE OK")
PY
```
Expected: prints shapes `[2, D]`, `[2, D]`, sim `[2, 2]`, then `SMOKE OK`. (Inspect `load_pretrained` log for sane missing/unexpected keys.)

- [ ] **Step 4: Commit**

```bash
git add lhp/model.py lhp/infer.py
git commit -m "feat(lhp): add LHPRetriever wrapper and inference API"
```

---

## Task 6: DDP training loop + config

**Files:**
- Create: `lhp/config.yaml`
- Create: `lhp/train.py`

**Interfaces:**
- Consumes: `LHPDataset`, `LHPTransform`, `LHPRetriever`, `tokenize_caption`, `beit3_utils.{get_rank,get_world_size}`.
- Produces: a runnable DDP training entry that saves `output_dir/lhp_epoch{N}.pth`.

**Note:** Smoke-verified via a short run; not a pytest unit test.

- [ ] **Step 1: Create `lhp/config.yaml`**

```yaml
image_root: 'data/PAB/'
train_file:
  - 'data/PAB/annotation/train/attr_0.json'
  # ... add attr_1.json ... attr_74.json (same list as configs/cmp.yaml)

resolution: 384
crop_scale: [0.5, 0.8]
local_prob: 0.5

eda: false
eda_p: 0.5
max_words: 56
max_tokens: 64

batch_size: 184          # effective target; reduce per-GPU + use grad-accum if OOM
lr: 1.0e-5
epochs: 3
weight_decay: 0.01
drop_path_rate: 0.1

beit3_ckpt: 'checkpoint/beit3_base_patch16_384_coco_retrieval.pth'
spm_model:  'checkpoint/beit3.spm'
output_dir: 'output/lhp'
```

- [ ] **Step 2: Implement `lhp/train.py`**

```python
import os
import argparse
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ruamel.yaml import YAML
from transformers import XLMRobertaTokenizer

from lhp.transform import LHPTransform
from lhp.dataset import LHPDataset
from lhp.model import LHPRetriever
from lhp.tokenization import tokenize_caption


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lhp/config.yaml")
    args = parser.parse_args()
    cfg = YAML(typ="safe").load(open(args.config))

    local_rank, rank, world = setup_ddp()          # BEFORE building model (ClipLoss rank/world)
    device = torch.device("cuda", local_rank)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tokenizer = XLMRobertaTokenizer(cfg["spm_model"])
    transform = LHPTransform(cfg["resolution"], tuple(cfg["crop_scale"]), cfg["local_prob"])
    dataset = LHPDataset(cfg["train_file"], cfg["image_root"], transform,
                         max_words=cfg["max_words"], eda=cfg["eda"], eda_p=cfg["eda_p"])

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    per_gpu_bs = cfg["batch_size"] // world
    loader = DataLoader(dataset, batch_size=per_gpu_bs, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = LHPRetriever(ckpt_path=cfg["beit3_ckpt"], drop_path_rate=cfg["drop_path_rate"]).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = math.ceil(len(dataset) / cfg["batch_size"]) * cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        for i, (image, captions, _path) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            toks = [tokenize_caption(tokenizer, c, cfg["max_tokens"]) for c in captions]
            text_ids = torch.tensor([t[0] for t in toks], device=device)
            padding_mask = torch.tensor([t[1] for t in toks], device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(image, text_ids, padding_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if rank == 0 and i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(loader)} loss {loss.item():.4f}")

        if rank == 0:
            torch.save({"model": model.module.state_dict(), "config": cfg},
                       os.path.join(cfg["output_dir"], f"lhp_epoch{epoch}.pth"))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run (short, requires prereqs)**

Use a tiny `train_file` (one small attr_*.json) and `epochs: 1`, then:
```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
  -m lhp.train --config lhp/config.yaml
```
Expected: distributed initializes; per-step loss prints and **trends down**; no hang at `dist.barrier()`; `output/lhp/lhp_epoch0.pth` written. (Loss printed differs from a single-process diagonal run → confirms cross-rank gather is active.)

- [ ] **Step 4: Commit**

```bash
git add lhp/config.yaml lhp/train.py
git commit -m "feat(lhp): add DDP contrastive training loop and config"
```

---

## Task 7: README for the module

**Files:**
- Create: `lhp/README.md`

- [ ] **Step 1: Write `lhp/README.md`** documenting: prerequisites (downloads + `sentencepiece`), how to run training (`torch.distributed.run` command), the no-hard-sampling / contrastive-per-paper constraints, batch/VRAM guidance (reduce per-GPU + grad-accum), and that output checkpoints feed a later rerank step.

- [ ] **Step 2: Commit**

```bash
git add lhp/README.md
git commit -m "docs(lhp): add module README"
```

---

## Self-Review (against spec)

- **§2 layout** → Tasks 1–7 create exactly the spec's files. ✓
- **§3.1 transform** → Task 2 (Inception normalize, RRC local / resize global, `local_prob`). ✓
- **§3.2 dataset** → Task 4 (no pose/hard, optional eda, returns image/caption/path). ✓
- **§3.3 model / ClipLoss / no idx** → Tasks 1+5 (built-in ClipLoss diagonal; no idx). ✓
- **§3.4 train / DDP / set_epoch / init-before-build** → Task 6. ✓
- **§3.5 infer** → Task 5. ✓
- **§6 prereqs** → Prerequisites block + Task 5/6 smoke gating. ✓
- **§7 batch/VRAM + DDP correctness** → Global Constraints + Task 6 (per-gpu bs, grad-accum note, gather verified in Task 1/6 smoke). ✓
- **§8 error handling** → Task 4 (corrupt image resample), Task 1 (strict-load logging via `load_state_dict`). ✓
- **§9 testing** → transform/tokenization/dataset = pytest (Tasks 2–4); model/DDP/infer = smoke (Tasks 5–6). ✓
- **No-hard-sampling** → enforced by using ClipLoss only; no hard fields anywhere. ✓
- **Type consistency** → `tokenize_caption(tokenizer,text,max_len)->(ids,mask)`, `LHPTransform.__call__->(tensor,view)`, `LHPRetriever.forward->(loss,v,l)` / `encode_image`/`encode_text`, `similarity(img,txt)` consistent across Tasks 3/2/5. ✓
```
