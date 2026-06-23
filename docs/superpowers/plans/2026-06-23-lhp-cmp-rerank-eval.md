# LHP→CMP Two-Stage Rerank Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lhp/rerank_eval.py` — a single-GPU 2-stage eval where the LHP dual-encoder selects top-k per query (stage-1) and the trained pose-aware CMP cross-encoder reranks those candidates (stage-2) on PAB test, reporting R@1/R@5/R@10, mAP, mINP via reused CMP scoring.

**Architecture:** Reuse CMP's `evaluation_itc`/`evaluation_itm`/`mAP` verbatim (identical rerank math) with `args.distributed=False`. Drive both stages from ONE `search_test_dataset` instance so gallery/query ordering aligns by index. LHP stage reuses `lhp.eval`'s encoders. First fix a timm-version import incompatibility so CMP and BeiT-3 coexist under timm 0.4.12.

**Tech Stack:** PyTorch (single GPU), CMP (`models.model_search.Search`, `eval.*`, `dataset.search_dataset`), BERT tokenizer; LHP (`lhp.eval`, `lhp.transform`, `lhp.infer`), `XLMRobertaTokenizer`.

## Global Constraints

- Reuse CMP `eval.{evaluation_itc, evaluation_itm, mAP}` verbatim; do not reimplement rerank math.
- Single GPU, no DDP: pass an `args` object with `.distributed = False` to `evaluation_itm`.
- Alignment is **index-only** (feature dims differ and never mix): correctness needs identical ordering → **no shuffle** + **same caption-flatten**. Drive both stages from one `search_test_dataset` instance.
- CMP reranker is pose-aware (`be_pose_img=True`) + no-hard (`be_hard=False`); stage-2 needs test pose images at `data_root/pose/<ann["image"]>` (present at `pose/test/`).
- `lhp_sims` shape must be `[N_query, N_gallery]` = `[len(test_ds.text), len(test_ds.image)]`.
- `lhp/rerank_eval.py` is an intentional CMP↔LHP bridge (imports CMP); keep it separate from the CMP-free `lhp/eval.py`.
- timm is pinned to **0.4.12** (BeiT-3 requirement); CMP imports must tolerate this.

**Prerequisites (user-managed):** trained CMP checkpoint (pose-aware, no-hard) + `configs/cmp.yaml` + `bert-base-uncased` tokenizer dir; test pose images at `data_root/pose/test/...`; LHP checkpoint + `beit3.spm`; `transformers` 4.x (for `XLMRobertaTokenizer(.spm)`).

---

## Task 1: timm-version-tolerant import in CMP swin

**Files:**
- Modify: `models/swin_transformer.py:15`

**Interfaces:**
- Produces: `models.model_search.Search` (and the whole CMP stack) importable under timm 0.4.12.

- [ ] **Step 1: Reproduce the failure**

Run: `.venv/bin/python -c "from models.model_search import Search"`
Expected: `ModuleNotFoundError: No module named 'timm.layers'` (timm 0.4.12 lacks `timm.layers`).

- [ ] **Step 2: Apply the tolerant import**

In `models/swin_transformer.py`, replace line 15:
```python
from timm.layers import DropPath, to_2tuple, trunc_normal_
```
with:
```python
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_          # timm >= 0.6
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_   # timm 0.4.12 (BeiT-3 pin)
```

- [ ] **Step 3: Verify CMP imports**

Run:
```bash
.venv/bin/python -c "from models.model_search import Search; from eval import evaluation_itc, evaluation_itm, mAP; print('CMP imports OK')"
```
Expected: `CMP imports OK` (no ModuleNotFoundError).

- [ ] **Step 4: Commit**

```bash
git add models/swin_transformer.py
git commit -m "fix(cmp): tolerant timm import in swin (works on pinned timm 0.4.12)"
```

---

## Task 2: rerank_eval helpers + unit tests

**Files:**
- Create: `lhp/rerank_eval.py` (imports + `eval_args` + `assert_aligned`)
- Test: `lhp/tests/test_rerank_eval.py`

**Interfaces:**
- Consumes: `lhp.infer.similarity`.
- Produces: `eval_args() -> SimpleNamespace(distributed=False)`; `assert_aligned(lhp_sims, n_query, n_gallery)` — raises `AssertionError` unless `lhp_sims.shape == (n_query, n_gallery)`.

- [ ] **Step 1: Write the failing test**

```python
# lhp/tests/test_rerank_eval.py
import torch
import pytest
from lhp.rerank_eval import eval_args, assert_aligned
from lhp.infer import similarity


def test_eval_args_distributed_false():
    assert eval_args().distributed is False


def test_assert_aligned_passes_then_raises():
    sims = torch.zeros(3, 5)          # [N_query=3, N_gallery=5]
    assert_aligned(sims, 3, 5)        # correct -> no raise
    with pytest.raises(AssertionError):
        assert_aligned(sims, 5, 3)    # swapped -> raise


def test_similarity_is_query_by_gallery():
    img = torch.randn(5, 8)           # 5 gallery
    txt = torch.randn(3, 8)           # 3 queries
    s = similarity(img, txt)          # txt @ img.T
    assert tuple(s.shape) == (3, 5)   # [N_query, N_gallery]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest lhp/tests/test_rerank_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'eval_args'` (or ModuleNotFound for `lhp.rerank_eval`).

- [ ] **Step 3: Create the top of `lhp/rerank_eval.py`**

```python
import argparse
from types import SimpleNamespace

import torch
from ruamel.yaml import YAML
from transformers import BertTokenizer, XLMRobertaTokenizer

from models.model_search import Search
from dataset import create_dataset, create_loader
from eval import evaluation_itc, evaluation_itm, mAP

from lhp.eval import load_retriever, encode_images, encode_texts
from lhp.transform import LHPTransform
from lhp.infer import similarity


def eval_args():
    """Minimal args object for CMP's evaluation_itm (single-GPU, no DDP)."""
    return SimpleNamespace(distributed=False)


def assert_aligned(lhp_sims, n_query, n_gallery):
    """Guard: stage-1 sims must be [N_query, N_gallery] so indices align with
    CMP's image_embeds/text_embeds/g_pids/q_pids."""
    assert tuple(lhp_sims.shape) == (n_query, n_gallery), (
        f"lhp_sims shape {tuple(lhp_sims.shape)} != (n_query={n_query}, n_gallery={n_gallery})"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest lhp/tests/test_rerank_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lhp/rerank_eval.py lhp/tests/test_rerank_eval.py
git commit -m "feat(lhp): rerank-eval helpers (eval_args, assert_aligned) + tests"
```

---

## Task 3: rerank_eval main (2-stage pipeline) + smoke

**Files:**
- Modify: `lhp/rerank_eval.py` (append `main`)

**Interfaces:**
- Consumes: `eval_args`, `assert_aligned` (Task 2); `load_retriever`, `encode_images`, `encode_texts` (lhp/eval.py); `similarity` (lhp/infer.py); `Search`, `create_dataset`, `create_loader`, `evaluation_itc`, `evaluation_itm`, `mAP` (CMP).

**Note:** Smoke-verified (needs CMP+LHP checkpoints + test pose + GPU + transformers 4.x); not a pytest unit test.

- [ ] **Step 1: Append `main` to `lhp/rerank_eval.py`**

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhp-config", default="lhp/config.yaml")
    ap.add_argument("--lhp-checkpoint", required=True)
    ap.add_argument("--lhp-kind", choices=["beit3", "lhp"], required=True)
    ap.add_argument("--cmp-config", default="configs/cmp.yaml")
    ap.add_argument("--cmp-checkpoint", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    with open(args.cmp_config) as f:
        cmp_cfg = YAML(typ="safe").load(f)
    cmp_cfg["load_params"] = False        # CMP checkpoint carries weights; skip swin/bert imagenet init
    with open(args.lhp_config) as f:
        lhp_cfg = YAML(typ="safe").load(f)

    # --- single ordered test source (pose-aware CMP dataset) ---
    _, test_ds = create_dataset(cmp_cfg, evaluate=True)
    cmp_loader = create_loader([test_ds], [None], batch_size=[cmp_cfg["batch_size_test"]],
                               num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    n_query, n_gallery = len(test_ds.text), len(test_ds.image)

    # --- STAGE 2 embeds (CMP, pose-fused) ---
    bert_tok = BertTokenizer.from_pretrained(cmp_cfg["text_encoder"])
    cmp_model = Search(config=cmp_cfg)
    cmp_model.load_pretrained(args.cmp_checkpoint)
    cmp_model = cmp_model.to(device).eval()
    _, image_embeds, text_embeds, text_atts = evaluation_itc(
        cmp_model, cmp_loader, bert_tok, device, cmp_cfg)   # discard CMP sims; keep CMP embeds

    # --- STAGE 1 sims (LHP), SAME order as test_ds ---
    lhp_model = load_retriever(args.lhp_kind, args.lhp_checkpoint, lhp_cfg["drop_path_rate"], device)
    spm_tok = XLMRobertaTokenizer(lhp_cfg["spm_model"])
    lhp_tf = LHPTransform(lhp_cfg["resolution"], tuple(lhp_cfg["crop_scale"]), local_prob=0.0)
    img_feats = encode_images(lhp_model, test_ds.ann, lhp_cfg["data_root"],
                              lhp_tf, lhp_cfg["batch_size_eval"], device)
    txt_feats = encode_texts(lhp_model, test_ds.text, spm_tok,
                             lhp_cfg["max_tokens"], lhp_cfg["batch_size_eval"], device)
    lhp_sims = similarity(img_feats, txt_feats)             # [N_query, N_gallery]
    assert_aligned(lhp_sims, n_query, n_gallery)

    # --- STAGE 2 rerank (reuse CMP), fed LHP sims for candidate selection ---
    score = evaluation_itm(cmp_model, device, cmp_cfg, eval_args(),
                           lhp_sims, image_embeds, text_embeds, text_atts)
    print(f"=== LHP({args.lhp_kind}) top-{cmp_cfg['k_test']} -> CMP rerank | "
          f"{n_gallery} gallery, {n_query} queries ===")
    mAP(score, test_ds.g_pids, test_ds.q_pids)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('lhp/rerank_eval.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 3: Smoke run (requires prereqs — both checkpoints + test pose + GPU + transformers 4.x)**

```bash
.venv/bin/python -m lhp.rerank_eval \
    --lhp-config lhp/config.yaml --lhp-checkpoint output/lhp/lhp_epoch2.pth --lhp-kind lhp \
    --cmp-config configs/cmp.yaml --cmp-checkpoint <cmp_checkpoint.pth>
```
Expected: prints a metrics table (R1/R5/R10/mAP/mINP). mAP should be **≥ the stage-1-only `lhp.eval` number** (rerank helps). The `assert_aligned` passes (no shape error).

- [ ] **Step 4: Commit**

```bash
git add lhp/rerank_eval.py
git commit -m "feat(lhp): 2-stage rerank eval (LHP top-k -> CMP cross-encoder) on PAB test"
```

---

## Self-Review (against spec)

- **§3 approach A (reuse evaluation_itc/itm/mAP, args.distributed=False)** → Task 3 `eval_args()` + calls. ✓
- **§4 data flow** (single dataset → CMP embeds → LHP sims → evaluation_itm → mAP) → Task 3 main. ✓
- **§5 alignment** (no shuffle: create_loader test + encode_images uses shuffle=False loader; same flatten: one `test_ds`) + shape guard → Task 2 `assert_aligned`, Task 3 `assert_aligned(lhp_sims, n_query, n_gallery)`. ✓
- **§2 pose-aware** → `create_dataset(cmp_cfg, evaluate=True)` builds pose-aware test_ds from `be_pose_img` in cmp_cfg; `evaluation_itc` fuses pose. ✓
- **§6 bridge file separate from lhp/eval.py** → new `lhp/rerank_eval.py`. ✓
- **timm conflict** (BeiT-3 pin vs CMP swin) → Task 1 tolerant import. ✓
- **§7 CLI** (`--lhp-*`, `--cmp-*`) → Task 3 argparse. ✓
- **Type consistency:** `similarity(img_feats, txt_feats)` == `[N_query, N_gallery]`; `assert_aligned(lhp_sims, len(test_ds.text), len(test_ds.image))`; `evaluation_itm(model, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts)` matches eval.py signature; `encode_images(model, ann, data_root, transform, bs, device)` / `encode_texts(model, captions, tokenizer, max_tokens, bs, device)` match lhp/eval.py. ✓
- **Reuse not reimplement:** mAP/evaluation_* imported from `eval`; encoders from `lhp.eval`. ✓
```
