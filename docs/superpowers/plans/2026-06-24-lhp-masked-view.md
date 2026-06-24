# LHP Masked-View (3rd transform) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a masked-image training view to LHP (local / masked / global, each 1/3) that masks 40–60% of image patches and blocks them from attention via a new `BEiT3ForRetrieval` subclass — without editing the vendored model.

**Architecture:** New subclass `MaskedBEiT3ForRetrieval` accepts a per-sample vision key-padding mask and routes the masked vision encode through `vision_embed`+`encoder` with `encoder_padding_mask` (masked patches get `-inf` attention + zeroed input; CLS unmasked). The transform emits a per-sample patch mask; the train loop assembles `[B, 1+num_patches]` and passes it through. `vision_padding_mask=None` ⇒ byte-identical to the parent (eval/baseline unaffected).

**Tech Stack:** PyTorch DDP, vendored BeiT-3 (`lhp/beit3`), torchscale encoder, timm 0.4.12.

## Global Constraints

- **New subclass only** — do NOT edit the vendored `lhp/beit3/modeling_finetune.py` `BEiT3ForRetrieval`.
- `vision_padding_mask=None` ⇒ identical to parent forward (eval/baseline path unchanged).
- CLS token (column 0) is **never masked**.
- Masked view is **train-only**: eval/rerank construct `LHPTransform` with `local_prob=0`, `masked_prob=0` → global-only; `LHPTransform.masked_prob` defaults to `0.0` so existing callers are unaffected.
- `patch_mask` is ALWAYS a bool tensor `[num_patches]` (all-`False` for local/global) — concretizes the spec's "None" so default DataLoader collation works.
- Mask ratio per masked sample: `random.uniform(0.4, 0.6)` of `num_patches=(resolution//16)**2` (=576 at 384).
- View pick by cumulative threshold: `<local_prob`→local, `<local_prob+masked_prob`→masked, else global. Train defaults: `local_prob=0.3333`, `masked_prob=0.3333`.
- No text masking, no hard sampling (augmentation only) → challenge-compliant.

---

## Task 1: MaskedBEiT3ForRetrieval subclass + mask helper + wiring

**Files:**
- Create: `lhp/masked_model.py`
- Modify: `lhp/beit3_loader.py` (build the subclass)
- Modify: `lhp/model.py` (`LHPRetriever.forward` passes `vision_padding_mask`)
- Test: `lhp/tests/test_masked_model.py`

**Interfaces:**
- Consumes: `lhp.beit3.modeling_finetune.BEiT3ForRetrieval`, `lhp.beit3.modeling_utils._get_base_config`.
- Produces: `MaskedBEiT3ForRetrieval(args)` with `forward(image, text_description, padding_mask, vision_padding_mask=None, only_infer=False)`; `build_vision_padding_mask(patch_mask: BoolTensor[B,N]) -> BoolTensor[B,1+N]`; `build_beit3_retrieval(drop_path_rate)` now returns a `MaskedBEiT3ForRetrieval`; `LHPRetriever.forward(image, text_ids, padding_mask, vision_padding_mask=None)`.

- [ ] **Step 1: Write the failing test**

```python
# lhp/tests/test_masked_model.py
import torch
from lhp.beit3_loader import build_beit3_retrieval
from lhp.masked_model import MaskedBEiT3ForRetrieval, build_vision_padding_mask


def test_build_returns_masked_subclass_with_cliploss():
    m = build_beit3_retrieval(drop_path_rate=0.0)
    assert isinstance(m, MaskedBEiT3ForRetrieval)
    assert type(m.criterion).__name__ == "ClipLoss"


def test_build_vision_padding_mask_prepends_false_cls_column():
    pm = torch.tensor([[True, False, True, False],
                       [False, False, False, True]])
    vpm = build_vision_padding_mask(pm)
    assert tuple(vpm.shape) == (2, 5)
    assert vpm[:, 0].tolist() == [False, False]   # CLS never masked
    assert torch.equal(vpm[:, 1:], pm)


def test_forward_parity_nomask_vs_allfalse_and_masked_backward():
    torch.manual_seed(0)
    m = build_beit3_retrieval(drop_path_rate=0.0).train()
    img = torch.randn(2, 3, 384, 384)
    txt = torch.randint(5, 64000, (2, 64))
    pad = torch.zeros(2, 64, dtype=torch.long)
    n_tokens = 1 + (384 // 16) ** 2                      # 577

    v_none = m(image=img, only_infer=True)[0]
    allfalse = torch.zeros(2, n_tokens, dtype=torch.bool)
    v_allfalse = m(image=img, vision_padding_mask=allfalse, only_infer=True)[0]
    assert torch.allclose(v_none, v_allfalse, atol=1e-5)  # mask path == parent when all-False

    vpm = torch.zeros(2, n_tokens, dtype=torch.bool); vpm[:, 1:300] = True   # mask some patches
    loss, _, _ = m(image=img, text_description=txt, padding_mask=pad, vision_padding_mask=vpm)
    assert loss.ndim == 0
    loss.backward()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_masked_model.py -v`
Expected: FAIL — `ModuleNotFoundError: lhp.masked_model`.

- [ ] **Step 3: Create `lhp/masked_model.py`**

```python
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
```

- [ ] **Step 4: Point `build_beit3_retrieval` at the subclass** — replace the body of `lhp/beit3_loader.py` with:

```python
from lhp.beit3.modeling_utils import _get_base_config
from lhp.beit3 import utils as beit3_utils
from lhp.masked_model import MaskedBEiT3ForRetrieval


def build_beit3_retrieval(drop_path_rate: float = 0.1):
    """Build MaskedBEiT3ForRetrieval (base, patch16, 384). Distributed must be
    initialized BEFORE calling so the internal ClipLoss gets correct rank/world_size.
    With vision_padding_mask=None it behaves identically to BEiT3ForRetrieval."""
    args = _get_base_config(img_size=384, drop_path_rate=drop_path_rate)
    return MaskedBEiT3ForRetrieval(args)


def load_pretrained(model, ckpt_path: str):
    """Load a BeiT-3 retrieval checkpoint with positional-embedding interpolation."""
    beit3_utils.load_model_and_may_interpolate(
        ckpt_path, model, model_key="model|module", model_prefix="")
    return model
```

- [ ] **Step 5: Thread `vision_padding_mask` through `LHPRetriever.forward`** — in `lhp/model.py` replace the `forward` method with:

```python
    def forward(self, image, text_ids, padding_mask, vision_padding_mask=None):
        # BEiT3ForRetrieval computes ClipLoss internally over in-batch (gathered) negatives
        return self.beit3(image=image, text_description=text_ids, padding_mask=padding_mask,
                          vision_padding_mask=vision_padding_mask)
```
(Leave `__init__`, `encode_image`, `encode_text` unchanged.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest lhp/tests/test_masked_model.py -v`
Expected: 3 passed. (Also run `.venv/bin/python -m pytest lhp/tests/test_import_model.py -v` → still 2 passed, since the subclass still has a ClipLoss criterion.)

- [ ] **Step 7: Commit**

```bash
git add lhp/masked_model.py lhp/beit3_loader.py lhp/model.py lhp/tests/test_masked_model.py
git commit -m "feat(lhp): MaskedBEiT3ForRetrieval subclass with vision key-padding mask"
```

---

## Task 2: LHPTransform masked branch

**Files:**
- Modify: `lhp/transform.py`
- Test: `lhp/tests/test_transform.py` (append cases)

**Interfaces:**
- Produces: `LHPTransform(resolution=384, crop_scale=(0.5,0.8), local_prob=0.5, masked_prob=0.0, mask_ratio_range=(0.4,0.6))`; `__call__(image) -> (tensor[3,res,res], view_str in {"local","masked","global"}, patch_mask: BoolTensor[num_patches])`. `patch_mask` all-`False` unless `view=="masked"`.

- [ ] **Step 1: Write the failing test (append to `lhp/tests/test_transform.py`)**

```python
import torch
from PIL import Image
from lhp.transform import LHPTransform


def _img2():
    return Image.new("RGB", (640, 480), (123, 117, 104))


def test_returns_three_tuple_with_patch_mask():
    t = LHPTransform(resolution=384, local_prob=1.0)   # always local
    out, view, patch_mask = t(_img2())
    assert tuple(out.shape) == (3, 384, 384) and view == "local"
    assert patch_mask.dtype == torch.bool and patch_mask.shape == (576,)
    assert patch_mask.sum().item() == 0                # local -> nothing masked


def test_masked_view_masks_40_to_60_percent():
    t = LHPTransform(resolution=384, local_prob=0.0, masked_prob=1.0)  # always masked
    for _ in range(10):
        out, view, patch_mask = t(_img2())
        assert view == "masked" and tuple(out.shape) == (3, 384, 384)
        frac = patch_mask.float().mean().item()
        assert 0.40 - 1e-6 <= frac <= 0.60 + 1e-6      # ratio in [0.4, 0.6]


def test_masked_prob_zero_never_masks():
    t = LHPTransform(resolution=384, local_prob=0.0, masked_prob=0.0)  # always global
    views = [t(_img2())[1] for _ in range(20)]
    assert set(views) == {"global"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_transform.py -v`
Expected: FAIL — `__call__` returns a 2-tuple (ValueError unpacking 3) / no `masked_prob`.

- [ ] **Step 3: Rewrite `lhp/transform.py`**

```python
import random
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation


class LHPTransform:
    """Stochastically pick local (random-resized-crop) / masked (full resize + patch
    attention-mask) / global (full resize) view, then ToTensor + Inception normalize.
    Returns (tensor, view_name, patch_mask) where patch_mask is a bool [num_patches]
    (all-False unless view == 'masked')."""

    def __init__(self, resolution=384, crop_scale=(0.5, 0.8), local_prob=0.5,
                 masked_prob=0.0, mask_ratio_range=(0.4, 0.6)):
        self.local_prob = local_prob
        self.masked_prob = masked_prob
        self.mask_ratio_range = mask_ratio_range
        self.num_patches = (resolution // 16) ** 2
        self._local = transforms.Compose([
            RandomResizedCropAndInterpolation(resolution, scale=crop_scale, interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
        ])
        self._global = transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC)
        self._finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    def _make_patch_mask(self, masked):
        patch_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        if masked:
            ratio = random.uniform(*self.mask_ratio_range)
            n_mask = int(round(self.num_patches * ratio))
            idx = torch.randperm(self.num_patches)[:n_mask]
            patch_mask[idx] = True
        return patch_mask

    def __call__(self, image):
        r = random.random()
        if r < self.local_prob:
            view, img = "local", self._local(image)
        elif r < self.local_prob + self.masked_prob:
            view, img = "masked", self._global(image)   # full image; patches blocked at model
        else:
            view, img = "global", self._global(image)
        return self._finalize(img), view, self._make_patch_mask(view == "masked")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest lhp/tests/test_transform.py -v`
Expected: all pass (the 3 original cases adapt — note the original cases unpack `(out, view)`; update those two assertions to unpack the 3-tuple if they fail, or they already ignore extra). If an original test breaks on unpacking, change its line `out, view = t(_img())` to `out, view, _ = t(_img())`.

- [ ] **Step 5: Commit**

```bash
git add lhp/transform.py lhp/tests/test_transform.py
git commit -m "feat(lhp): add masked view to LHPTransform (3-way local/masked/global)"
```

---

## Task 3: Wire patch mask through dataset + train + config (smoke)

**Files:**
- Modify: `lhp/dataset.py` (return `patch_mask`)
- Modify: `lhp/train.py` (assemble + pass `vision_padding_mask`)
- Modify: `lhp/config.yaml` (`local_prob`, `masked_prob`)

**Interfaces:**
- Consumes: `LHPTransform` 3-tuple (Task 2), `LHPRetriever.forward(..., vision_padding_mask)` + `build_vision_padding_mask` (Task 1).

**Note:** smoke-verified (needs weights + GPU); not a pytest unit test.

- [ ] **Step 1: `lhp/dataset.py` — return the patch mask**

In `__getitem__`, change the unpack and the return:
```python
                image, _view, patch_mask = self.transform(image)
                caption = pre_caption(ann["caption"], self.max_words, self.eda, self.eda_p)
                return image, caption, ann["image"], patch_mask
```
(Everything else — the `.jpg`→`.webp` swap, the resample loop — unchanged.)

- [ ] **Step 2: `lhp/train.py` — build and pass `vision_padding_mask`**

Add import:
```python
from lhp.masked_model import build_vision_padding_mask
```
Change the transform construction to include `masked_prob`:
```python
    transform = LHPTransform(cfg["resolution"], tuple(cfg["crop_scale"]),
                             cfg["local_prob"], cfg["masked_prob"])
```
Change the batch loop header + mask assembly + model call:
```python
        for i, (image, captions, _path, patch_mask) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            vision_padding_mask = build_vision_padding_mask(patch_mask.to(device))  # [B, 1+num_patches]
            toks = [tokenize_caption(tokenizer, c, cfg["max_tokens"]) for c in captions]
            text_ids = torch.tensor([t[0] for t in toks], device=device)
            padding_mask = torch.tensor([t[1] for t in toks], device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(image, text_ids, padding_mask, vision_padding_mask)
```

- [ ] **Step 3: `lhp/config.yaml` — set the 3-way split**

Ensure these keys (add `masked_prob`, set `local_prob`):
```yaml
local_prob: 0.3333
masked_prob: 0.3333
```
(global gets the remaining ~1/3.)

- [ ] **Step 4: Verify it parses + config loads**

```bash
.venv/bin/python -c "import ast; ast.parse(open('lhp/train.py').read()); ast.parse(open('lhp/dataset.py').read()); print('parse OK')"
.venv/bin/python -c "from ruamel.yaml import YAML; c=YAML(typ='safe').load(open('lhp/config.yaml')); print('local', c['local_prob'], 'masked', c['masked_prob'])"
```
Expected: `parse OK`; `local 0.3333 masked 0.3333`.

- [ ] **Step 5: Smoke run (requires weights + GPU + transformers 4.x)**

Use 1 small `train_file` + `epochs: 1`:
```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
  -m lhp.train --config lhp/config.yaml
```
Expected: per-step loss prints and trends down; checkpoint saved. (~1/3 of samples now go through the masked attention path.)

- [ ] **Step 6: Commit**

```bash
git add lhp/dataset.py lhp/train.py lhp/config.yaml
git commit -m "feat(lhp): wire masked-view patch mask through dataset/train/config"
```

---

## Self-Review (against spec)

- **§2 attention-block via key_padding_mask** → Task 1 routes masked vision encode through `encoder(encoder_padding_mask=...)`. ✓
- **§3 new subclass, no edit to vendored BEiT3ForRetrieval** → Task 1 `lhp/masked_model.py`; vendored file untouched; `build_beit3_retrieval` switched to subclass (strict superset). ✓
- **§4 components** → Task 1 (masked_model, beit3_loader, model), Task 2 (transform), Task 3 (dataset, train, config). ✓
- **§5 data flow** (transform patch_mask → batch [B,577] → model) → Task 2 + Task 3 + `build_vision_padding_mask`. ✓
- **§6 CLS never masked / train-only / no text mask** → `build_vision_padding_mask` CLS col False; `masked_prob` defaults 0 (eval unaffected); only vision masked. ✓
- **§7 probabilities** (cumulative thresholds, 1/3 each) → Task 2 `__call__`, Task 3 config. ✓
- **§8 testing** → Task 1 (build/parity/mask-helper/backward), Task 2 (transform/ratio), Task 3 (smoke). ✓
- **Parent-parity when mask None/all-False** → Task 1 `test_forward_parity_nomask_vs_allfalse...`. ✓
- **eval/baseline unaffected** → `vision_padding_mask` defaults None; `masked_prob` defaults 0; `lhp/eval.py` keeps `transform(img)[0]`. ✓
- **Type consistency:** `build_vision_padding_mask([B,N])->[B,1+N]`; transform `->(tensor,view,patch_mask[num_patches])`; `LHPRetriever.forward(image,text_ids,padding_mask,vision_padding_mask)`; `MaskedBEiT3ForRetrieval.forward(image,text_description,padding_mask,vision_padding_mask,only_infer)`. ✓
```
