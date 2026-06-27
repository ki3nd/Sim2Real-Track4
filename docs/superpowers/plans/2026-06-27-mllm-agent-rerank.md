# MLLM Agent Rerank (SSDC Stage-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `mllm_rerank/` module that reranks the repo's CMP Stage-1 retrieval results with a single-pass Detective→Analyst→Writer MLLM agent chain (Qwen3-VL-8B-Instruct, zero-shot, vLLM), fusing the CMP structural score with a semantic cosine `cos(E_txt(T), E_txt(T_new))` (SSDC paper Eq. 2-4).

**Architecture:** For each text query, take the top-k image candidates by CMP ITM score (`S_str`), gate on `xi`, run the 3-agent chain per surviving candidate to produce a new caption `T_new`, score `S_sem = cosine(query, T_new)` via the frozen CMP BERT text tower, fuse `S_final = λ·S_str + (1-λ)·S_sem` on processed positions, then rank. All reused SSDC logic is **ported (copied + adapted)** into `mllm_rerank/` — never imported across `open-sources/`.

**Tech Stack:** Python, PyTorch, vLLM, transformers (`AutoProcessor`, `BertTokenizer`), `qwen_vl_utils`, `ruamel.yaml`, `prettytable`, pytest. Reuses repo's `models.model_search.Search`, `models.cmp` methods, `dataset.create_dataset/create_loader`.

## Global Constraints

- **No cross-repo import:** no file under `mllm_rerank/` may contain the string `open-sources` or add any path to `sys.path`. All SSDC logic is copied in, not imported. (Test-enforced in Task 6.)
- **Zero-shot MLLM only:** Qwen3-VL-8B-**Instruct**, no fine-tuning, no LoRA loading, no hard-negative mining, no training of any model.
- **Semantic-cosine scoring (paper), not hard-boost:** rerank uses `S_sem = cos(E_txt(T), E_txt(T_new))`, never the released-SSDC `sims_[i][idx]=1.0` shortcut.
- **Frozen CMP BERT text tower** embeds both `T` and `T_new` (same space as `S_str`).
- **Defaults:** `lambda=0.4`, `xi=0.1`, `top_k=10` — all overridable via `mllm_rerank/config.yaml`.
- **Imports allowed:** this repo (`models.*`, `dataset.*`) + third-party only.
- Tests are plain pytest functions under `tests/` (match existing `tests/test_*.py` style — no classes, no fixtures needed).

---

### Task 1: Package scaffold + prompts + parsing helpers

**Files:**
- Create: `mllm_rerank/__init__.py`
- Create: `mllm_rerank/prompts.py`
- Test: `tests/test_mllm_prompts.py`

**Interfaces:**
- Produces:
  - `DETECTIVE_PROMPT: str`, `ANALYST_PROMPT: str`, `WRITER_PROMPT: str` (module-level strings; `DETECTIVE_PROMPT` has a `{cap}` field, `WRITER_PROMPT` has a `{cap}` field, `ANALYST_PROMPT` has none).
  - `process_checklist(raw_answers: list[str]) -> list[list[str]]` — clean a batch of raw Analyst answers into per-sample lists of sentences.
  - `parse_writer_caption(raw: str) -> str` — extract the `"caption"` value from the Writer's JSON output; fall back to the raw text (stripped) if JSON parsing fails or the key is absent.

- [ ] **Step 1: Create the package marker**

`mllm_rerank/__init__.py`:

```python
"""MLLM agent rerank (SSDC Stage-2): Detective -> Analyst -> Writer, semantic-cosine fusion."""
```

- [ ] **Step 2: Write the failing test**

`tests/test_mllm_prompts.py`:

```python
from mllm_rerank.prompts import (
    DETECTIVE_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    process_checklist,
    parse_writer_caption,
)


def test_prompts_have_expected_fields():
    assert "{cap}" in DETECTIVE_PROMPT
    assert "{cap}" in WRITER_PROMPT
    assert "{cap}" not in ANALYST_PROMPT  # Analyst takes only the image
    assert "Yes" in DETECTIVE_PROMPT and "No" in DETECTIVE_PROMPT


def test_process_checklist_cleans_indexed_answers():
    raw = (
        "1.Yes, the person is male.\n"
        "2.The person has short black hair.\n"
        "3.He is wearing a red t-shirt.\n"
    )
    out = process_checklist([raw])
    assert len(out) == 1
    sents = out[0]
    # index "N." stripped, "Yes, " stripped, capitalized, trailing period
    assert "The person is male." in sents
    assert "The person has short black hair." in sents
    assert all(s == "" or s.endswith(".") for s in sents)
    # no leading numeric index survives
    assert not any(s[:2] in ("1.", "2.", "3.") for s in sents if s)


def test_process_checklist_drops_colon_header_lines():
    raw = "Answers:\n1.The person is female.\n"
    out = process_checklist([raw])
    assert "The person is female." in out[0]
    assert "Answers:" not in out[0]


def test_parse_writer_caption_valid_json():
    raw = '{"caption": "A man in a red shirt is falling down."}'
    assert parse_writer_caption(raw) == "A man in a red shirt is falling down."


def test_parse_writer_caption_falls_back_on_bad_json():
    raw = "A woman walking a dog."
    assert parse_writer_caption(raw) == "A woman walking a dog."


def test_parse_writer_caption_extracts_json_embedded_in_text():
    raw = 'Output: {"caption": "Two people fighting."} done'
    assert parse_writer_caption(raw) == "Two people fighting."
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_prompts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.prompts'`.

- [ ] **Step 4: Implement `mllm_rerank/prompts.py`**

The three prompts are copied verbatim from SSDC `vllm_infer_SSDC.py::round_llm`. `process_checklist` ports `process_cap_`; `parse_writer_caption` is new (paper needs the caption text, not a state flag).

```python
"""Detective / Analyst / Writer prompts and output parsers (ported from SSDC)."""
import json
import re

DETECTIVE_PROMPT = """You are an expert in Person Re-identification and Anomaly Detection.
Task: Determine if the text accurately describes the **primary person** in the image.

Evaluation Criteria:
1.**Appearance**: Check gender, clothing color, clothing type (upper/lower), and distinct accessories.
2.**Action**: Check if the described action (e.g., walking, falling, fighting) matches the person's behavior.
3.**Ignore**: Do not judge based on background details or lighting differences unless they obscure the person.

Text: {cap}

Does this text accurately describe the image? Answer STRICTLY with "Yes" or "No"."""

ANALYST_PROMPT = """According to the pedestrian image, answer the following questions one by one:

1.The person is male or female?
2.What hairstyle does the person have, such as hair length and color?
3.What is this person wearing on his upper body? If clearly visible, what are the color, type, and sleeve length?
4.What are the characteristics of this person's pants? If clearly visible, what are the color, type, and trouser leg length?
5.Does this person have any patterns on his/her clothes or pants?
6.What are the characteristics of this person's shoes? If clearly visible, what are the color and style?
7.Does this person wear glasses? If clearly visible, what are the color and style?
8.Is this person wearing a scarf? If clearly visible, what are the color and style?
9.Does this person have something in his/her hand? If so, what is it and what color is it?
10.Does this person carry a backpack? If clearly visible, what are the color and style?
11.Does this person wear a hat? If clearly visible, what are the color and style?
12.Is this person wearing a belt or waistband?
13.What is this person doing?
14.What is the background?
15.Are there other people in the background of this person?"""

WRITER_PROMPT = """
    Task: Aggregate the following subtexts into a single continuous and concise text paragraph.
    Format: Return the result strictly as a JSON object with a single key "caption".

    Requirements:

    1.Grammar Flow: Ensure the transition after the prefix is natural.
    2.Keep the output concise, fluent and grammatical.
    3.The final returned content must be a JSON object with the single key "caption" whose value is the aggregated caption string.

    Now let's get started.
    Subtexts: {cap}
    Output:
    """


def process_checklist(raw_answers):
    """Clean raw Analyst answers into per-sample lists of sentences (ports SSDC process_cap_)."""
    tmps = []
    for c in raw_answers:
        c = c.split('\n')
        tmp = []
        for cc in c:
            if ': ' in cc:
                continue
            try:
                cc = cc.split('.')[1]
                if 'Yes, ' in cc:
                    cc = cc.replace('Yes, ', '')
                if 'No, ' in cc:
                    cc = cc.replace('No, ', '')
                cc = cc[:1].upper() + cc[1:]
                if cc[-1:] != '.':
                    cc += '.'
            except Exception:
                cc = ''
            tmp.append(cc)
        tmps.append(tmp)
    return tmps


def parse_writer_caption(raw):
    """Extract the "caption" value from the Writer's JSON; fall back to stripped raw text."""
    if raw is None:
        return ""
    text = raw.strip()
    # try direct JSON parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "caption" in obj:
            return str(obj["caption"]).strip()
    except Exception:
        pass
    # try to find an embedded {...} block containing "caption"
    match = re.search(r'\{.*?"caption".*?\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "caption" in obj:
                return str(obj["caption"]).strip()
        except Exception:
            pass
    return text
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_prompts.py -q`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add mllm_rerank/__init__.py mllm_rerank/prompts.py tests/test_mllm_prompts.py
git commit -m "feat(mllm-rerank): package scaffold + Detective/Analyst/Writer prompts + parsers"
```

---

### Task 2: Metrics helpers (ported)

**Files:**
- Create: `mllm_rerank/metrics.py`
- Test: `tests/test_mllm_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `rank(similarity: Tensor[Q,G], q_pids: Tensor[Q], g_pids: Tensor[G], max_rank=10, get_mAP=True)` → `(all_cmc, mAP, mINP, indices)` when `get_mAP`, else `(all_cmc, indices)`. `all_cmc` is a length-`max_rank` float tensor of CMC percentages.
  - `get_metrics(similarity, qids, gids, n_, retur_indices=False)` → a 7-element list `[n_, R1, R5, R10, mAP, mINP, rSum]` (numpy floats), or `(list, indices)` if `retur_indices`.
  - `print_rs(sims_dict: dict[str, Tensor], qids, pids, logger)` → prints a PrettyTable of metrics for each key.

- [ ] **Step 1: Write the failing test**

`tests/test_mllm_metrics.py`:

```python
import torch
from mllm_rerank.metrics import rank, get_metrics


def test_perfect_ranking_gives_r1_100():
    # 3 queries, 3 gallery; identity => each query's match is rank-1
    sim = torch.eye(3)
    qids = torch.tensor([0, 1, 2])
    gids = torch.tensor([0, 1, 2])
    all_cmc, mAP, mINP, indices = rank(sim, qids, gids, max_rank=3, get_mAP=True)
    assert float(all_cmc[0]) == 100.0
    assert float(mAP) == 100.0


def test_get_metrics_row_shape_and_r1():
    sim = torch.eye(5)
    qids = torch.tensor([0, 1, 2, 3, 4])
    gids = torch.tensor([0, 1, 2, 3, 4])
    row = get_metrics(sim, qids, gids, "unit-t2i", retur_indices=False)
    assert len(row) == 7
    assert row[0] == "unit-t2i"
    assert round(float(row[1]), 2) == 100.00  # R1
    # rSum == R1+R5+R10
    assert round(float(row[6]), 2) == round(float(row[1]) + float(row[2]) + float(row[3]), 2)


def test_worst_ranking_lowers_r1():
    # anti-diagonal: correct match is always last => R1 should be 0
    sim = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    qids = torch.tensor([0, 1])
    gids = torch.tensor([0, 1])
    all_cmc, mAP, mINP, indices = rank(sim, qids, gids, max_rank=2, get_mAP=True)
    assert float(all_cmc[0]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.metrics'`.

- [ ] **Step 3: Implement `mllm_rerank/metrics.py`** (ported verbatim from SSDC `vllm_infer_SSDC.py`)

```python
"""Retrieval metrics (R@K / mAP / mINP) — ported from SSDC vllm_infer_SSDC.py."""
import torch
from prettytable import PrettyTable


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.)
           for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    row = [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
           t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]
    if retur_indices:
        return row, indices
    return row


def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
    for key in sims_dict.keys():
        rs = get_metrics(sims_dict[key], qids, pids, f'{key}-t2i', False)
        table.add_row(rs)
    for col in ["R1", "R5", "R10", "mAP", "mINP", "rSum"]:
        table.custom_format[col] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))
```

Note: the `mINP` line indexes `match_row.nonzero()[-1]`; the unit tests use galleries where every query has a match, so this is safe. (Mirrors SSDC, which assumes the same.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_metrics.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/metrics.py tests/test_mllm_metrics.py
git commit -m "feat(mllm-rerank): port R@K/mAP/mINP metrics helpers from SSDC"
```

---

### Task 3: CMP Stage-1 feature/score helpers (ported, repo `Search`)

**Files:**
- Create: `mllm_rerank/cmp_features.py`
- Test: `tests/test_mllm_cmp_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `load_cmp_components(config_path: str, checkpoint_path: str, device) -> (config: dict, tokenizer, model)` — loads YAML, builds repo `models.model_search.Search`, calls `model.load_pretrained(checkpoint_path)`, `BertTokenizer.from_pretrained(config['text_encoder'])`.
  - `extract_cmp_features(model, data_loader, tokenizer, device, config) -> dict` with keys `sims_itc, text_feats, image_feats, text_embeds, image_embeds, text_atts, img_paths, texts`.
  - `compute_cmp_itm_scores(model, features: dict, device, config) -> Tensor[Q,G]` — min-max-normalized ITM rerank matrix (`S_str`).
  - `embed_texts(model, tokenizer, texts: list[str], device, config) -> Tensor[N, D]` — L2-normalized CMP text features for both `T` and `T_new`.

- [ ] **Step 1: Write the failing test** (mock `Search` model — no GPU/weights)

`tests/test_mllm_cmp_features.py`:

```python
import torch
import torch.nn.functional as F
from mllm_rerank.cmp_features import embed_texts


class _FakeTok:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        n = len(texts)

        class _Enc:
            input_ids = torch.zeros(n, max_length, dtype=torch.long)
            attention_mask = torch.ones(n, max_length, dtype=torch.long)

            def to(self, device):
                return self
        return _Enc()


class _FakeModel:
    """Mimics Search.get_text_embeds / get_text_feat: maps each row to a fixed direction."""
    def get_text_embeds(self, input_ids, attention_mask):
        b = input_ids.size(0)
        return torch.arange(1, b + 1, dtype=torch.float).view(b, 1, 1).repeat(1, 4, 8)

    def get_text_feat(self, text_embed):
        # use the [:,0] token -> [b, 8]
        return text_embed[:, 0, :]


def test_embed_texts_returns_l2_normalized_rows():
    model, tok = _FakeModel(), _FakeTok()
    cfg = {"max_tokens": 56}
    out = embed_texts(model, tok, ["a", "b", "c"], device=torch.device("cpu"), config=cfg)
    assert out.shape == (3, 8)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)  # L2-normalized
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_cmp_features.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.cmp_features'`.

- [ ] **Step 3: Implement `mllm_rerank/cmp_features.py`**

Ported from SSDC `load_cmp_components` / `extract_cmp_features` / `compute_cmp_itm_scores`, but importing **this repo's** `Search` and `dataset` (no `CMP_ROOT`/`sys.path`). `config['text_encoder']` and any `*_config` paths are used as-is (relative to repo root, where `rerank.py` runs).

```python
"""CMP Stage-1 feature extraction + ITM scoring + text embedding (ported from SSDC)."""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ruamel.yaml import YAML
from transformers import BertTokenizer

from models.model_search import Search

yaml = YAML(typ="safe")


def load_cmp_components(config_path, checkpoint_path, device):
    config = yaml.load(open(config_path, 'r'))
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return config, tokenizer, model


@torch.no_grad()
def extract_cmp_features(model, data_loader, tokenizer, device, config):
    model.eval()
    texts = data_loader.dataset.text
    text_bs = config.get('batch_size_test_text', 256)
    text_feats_list, text_embeds_list, text_atts_list = [], [], []

    for i in tqdm(range(0, len(texts), text_bs), desc="Text Features"):
        batch_text = texts[i:i + text_bs]
        text_input = tokenizer(
            batch_text, padding='max_length', truncation=True,
            max_length=config['max_tokens'], return_tensors="pt",
        ).to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)
        text_embeds_list.append(text_embed.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        text_feats_list.append(F.normalize(text_feat, dim=-1).cpu())

    image_feats_list, image_embeds_list, img_paths = [], [], []
    for image, pose, img_idx in tqdm(data_loader, desc="Image Features"):
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        if config.get('be_pose_img', False) and pose is not None:
            pose = pose.to(device)
            pose_embed, _ = model.get_vision_embeds(model.pose_conv(pose))
            image_embed = model.pose_block(image_embed, pose_embed)
        image_feat = model.get_image_feat(image_embed)
        image_embeds_list.append(image_embed.cpu())
        image_feats_list.append(F.normalize(image_feat, dim=-1).cpu())
        for idx in img_idx:
            path = data_loader.dataset.ann[idx.item()]['image']
            img_paths.append(os.path.join(data_loader.dataset.image_root, path))

    return {
        'sims_itc': torch.cat(text_feats_list, dim=0) @ torch.cat(image_feats_list, dim=0).t(),
        'text_feats': torch.cat(text_feats_list, dim=0),
        'image_feats': torch.cat(image_feats_list, dim=0),
        'text_embeds': torch.cat(text_embeds_list, dim=0),
        'image_embeds': torch.cat(image_embeds_list, dim=0),
        'text_atts': torch.cat(text_atts_list, dim=0),
        'img_paths': img_paths,
        'texts': texts,
    }


@torch.no_grad()
def compute_cmp_itm_scores(model, features, device, config):
    sims_matrix = features['sims_itc'].to(device)
    image_embeds = features['image_embeds'].to(device)
    text_embeds = features['text_embeds'].to(device)
    text_atts = features['text_atts'].to(device)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0, device=device)
    k_test = config.get('k_test', 128)

    for i, sims in enumerate(tqdm(sims_matrix, desc="ITM Re-ranking")):
        topk_sim, topk_idx = sims.topk(k=min(k_test, sims.size(0)), dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        current_text_embed = text_embeds[i].unsqueeze(0).repeat(encoder_output.size(0), 1, 1)
        current_text_att = text_atts[i].unsqueeze(0).repeat(encoder_output.size(0), 1)
        output = model.get_cross_embeds(
            encoder_output, encoder_att,
            text_embeds=current_text_embed, text_atts=current_text_att,
        )[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    mask = score_matrix_t2i == 1000.0
    score_matrix_t2i[mask] = replacement[mask]

    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_sim = (sims_matrix - sims_matrix.min()) / (sims_matrix.max() - sims_matrix.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim
    return score_matrix_t2i.cpu()


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, config):
    """L2-normalized CMP text features for an arbitrary list of strings (T or T_new)."""
    max_len = config.get('max_tokens_text', config.get('max_tokens', 56))
    text_input = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_len, return_tensors="pt",
    ).to(device)
    text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_feat = model.get_text_feat(text_embed)
    return F.normalize(text_feat, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_cmp_features.py -q`
Expected: PASS (1 passed). (Note: `import torch` triggers loading repo `models.model_search`; if that import itself errors in the dev env, the test still validates `embed_texts` logic once imports resolve — the implementer confirms the module imports cleanly.)

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/cmp_features.py tests/test_mllm_cmp_features.py
git commit -m "feat(mllm-rerank): port CMP feature/ITM helpers + embed_texts (repo Search, no cross-import)"
```

---

### Task 4: vLLM MLLM wrapper (ported)

**Files:**
- Create: `mllm_rerank/mllm.py`
- Test: `tests/test_mllm_wrapper.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class MLLMs(model_dir, gpu_memory_utilization=0.7, max_model_len=1536)`: lazily imports `vllm`/`qwen_vl_utils` inside `__init__` (so the module imports without vLLM installed).
    - `generate_response_multi_images(questions: list[str], images: list, sys="You are a helpful assistant.", t=0.01) -> list[str]` (one image per prompt).
    - `generate_response_text(questions: list[str], sys="You are a helpful assistant.", t=0.01) -> list[str]` (text-only).
  - `batch_infer(llm, b_prompts: list[str], images: list, micro_batch=8, t=0.01) -> list[str]`.
  - `batch_infer_txt(llm, b_prompts: list[str], micro_batch=16, t=0.01) -> list[str]`.

- [ ] **Step 1: Write the failing test** (no vLLM — test batching logic with a fake llm)

`tests/test_mllm_wrapper.py`:

```python
import ast
from mllm_rerank.mllm import batch_infer, batch_infer_txt


class _FakeLLM:
    def generate_response_multi_images(self, questions, images, t=0.01):
        return [f"img:{q}" for q in questions]

    def generate_response_text(self, questions, t=0.01):
        return [f"txt:{q}" for q in questions]


def test_batch_infer_preserves_order_and_count():
    prompts = [f"p{i}" for i in range(20)]
    images = [None] * 20
    out = batch_infer(_FakeLLM(), prompts, images, micro_batch=8, t=0.01)
    assert out == [f"img:p{i}" for i in range(20)]


def test_batch_infer_txt_preserves_order_and_count():
    prompts = [f"q{i}" for i in range(35)]
    out = batch_infer_txt(_FakeLLM(), prompts, micro_batch=16, t=0.01)
    assert out == [f"txt:q{i}" for i in range(35)]


def test_module_has_no_cross_repo_import():
    with open("mllm_rerank/mllm.py") as f:
        src = f.read()
    assert "open-sources" not in src
    assert "sys.path" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_wrapper.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.mllm'`.

- [ ] **Step 3: Implement `mllm_rerank/mllm.py`**

Ported from SSDC `MLLMs` + `batch_infer`/`batch_infer_txt`. vLLM and `qwen_vl_utils` imports are **deferred into `__init__`** so the module (and its batching helpers) import without vLLM present.

```python
"""Qwen3-VL-8B vLLM wrapper + batched inference (ported from SSDC vllm_infer_SSDC.py)."""
import os
import gc
import torch
from tqdm import tqdm


class MLLMs(object):
    def __init__(self, model_dir, gpu_memory_utilization=0.7, max_model_len=1536):
        from vllm import LLM, SamplingParams           # deferred
        from transformers import AutoProcessor
        self._SamplingParams = SamplingParams
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True,max_split_size_mb:256")
        self.model_dir = model_dir
        self.llm = LLM(
            model=model_dir, tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=256, max_model_len=max_model_len,
            enforce_eager=True, disable_log_stats=True, trust_remote_code=True,
            limit_mm_per_prompt={"image": 1}, enable_chunked_prefill=True,
            max_num_batched_tokens=4096, dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def generate_response_multi_images(self, questions, images=None,
                                       sys="You are a helpful assistant.", t=0.01):
        from qwen_vl_utils import process_vision_info  # deferred
        try:
            messages = [
                [{"role": "system", "content": sys},
                 {"role": "user", "content": [
                     {"type": "image", "image": images[i], "min_pixels": 50176, "max_pixels": 50176},
                     {"type": "text", "text": p}]}]
                for i, p in enumerate(questions)]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            image_data = [process_vision_info(m)[0] for m in messages]
            inputs = [{"prompt": p, "multi_modal_data": {"image": image_data[i]}}
                      for i, p in enumerate(prompts)]
            sp = self._SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sp)
            results = [o.outputs[0].text for o in outputs]
            del inputs, outputs, image_data, messages, prompts
            torch.cuda.empty_cache(); gc.collect()
            return results
        except Exception as e:
            print(f"Generation failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            return [""] * len(questions)

    def generate_response_text(self, questions, sys="You are a helpful assistant.", t=0.01):
        try:
            messages = [
                [{"role": "system", "content": sys},
                 {"role": "user", "content": [{"type": "text", "text": p}]}]
                for p in questions]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            inputs = [{"prompt": p} for p in prompts]
            sp = self._SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sp)
            results = [o.outputs[0].text for o in outputs]
            del inputs, outputs, messages, prompts
            torch.cuda.empty_cache(); gc.collect()
            return results
        except Exception as e:
            print(f"Text generation failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            return [""] * len(questions)


def batch_infer(llm, b_prompts, images, micro_batch=8, t=0.01):
    results = []
    n = len(b_prompts)
    n_batches = (n - 1) // micro_batch + 1 if n else 0
    for i in tqdm(range(n_batches), desc="Batch Inference"):
        start, end = i * micro_batch, min((i + 1) * micro_batch, n)
        try:
            rs = llm.generate_response_multi_images(
                questions=b_prompts[start:end], images=images[start:end], t=t)
            results += rs
        except Exception as e:
            print(f"Batch {i+1} failed: {e}")
            results += [""] * (end - start)
    return results


def batch_infer_txt(llm, b_prompts, micro_batch=16, t=0.01):
    results = []
    n = len(b_prompts)
    n_batches = (n - 1) // micro_batch + 1 if n else 0
    for i in tqdm(range(n_batches), desc="Text Inference"):
        start, end = i * micro_batch, min((i + 1) * micro_batch, n)
        try:
            rs = llm.generate_response_text(questions=b_prompts[start:end], t=t)
            results += rs
        except Exception as e:
            print(f"Text batch {i+1} failed: {e}")
            results += [""] * (end - start)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_wrapper.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/mllm.py tests/test_mllm_wrapper.py
git commit -m "feat(mllm-rerank): vLLM Qwen3-VL wrapper + batched inference (deferred vllm import)"
```

---

### Task 5: Single-pass Detective Squad orchestration

**Files:**
- Create: `mllm_rerank/squad.py`
- Test: `tests/test_mllm_squad.py`

**Interfaces:**
- Consumes: `mllm_rerank.prompts` (`DETECTIVE_PROMPT`, `ANALYST_PROMPT`, `WRITER_PROMPT`, `process_checklist`, `parse_writer_caption`); `mllm_rerank.mllm` (`batch_infer`, `batch_infer_txt`).
- Produces:
  - `load_image(path: str)` → PIL RGB image (helper).
  - `run_squad(llm, queries: list[str], cand_img_paths: list[list[str]], gate_mask: list[bool], image_micro_batch=8, text_micro_batch=16, temperature=0.01) -> dict[tuple[int,int], str]`
    - `queries[i]` = the i-th query text `T_i`.
    - `cand_img_paths[i]` = list of top-k image **paths** for query i (length k).
    - `gate_mask[i]` = whether query i passed the `xi` structural gate (False ⇒ skip entirely).
    - Returns `{(i, j): T_new}` for every `(query i, candidate j)` the **Detective** answered "Yes" (j is the index into `cand_img_paths[i]`). Candidates answered "No" (or empty) are absent from the dict.
    - Internally batches all gated (i,j) Detective calls together, then all surviving Analyst calls, then all surviving Writer calls (single pass, 3 batched stages).

- [ ] **Step 1: Write the failing test** (mock llm — no vLLM)

`tests/test_mllm_squad.py`:

```python
from mllm_rerank.squad import run_squad


class _ScriptedLLM:
    """Detective says Yes for candidate j==0 only; Analyst returns a checklist; nothing else needed."""
    def __init__(self):
        self.calls = {"img": 0, "txt": 0}

    def generate_response_multi_images(self, questions, images, t=0.01):
        self.calls["img"] += 1
        out = []
        for q in questions:
            if "STRICTLY" in q:                 # Detective prompt
                # the test encodes the candidate index in the image placeholder
                out.append("Yes" if images[len(out)] == "cand0" else "No")
            else:                               # Analyst prompt
                out.append("1.The person is male.\n2.Short black hair.\n")
        return out

    def generate_response_text(self, questions, t=0.01):
        self.calls["txt"] += 1
        return ['{"caption": "A man with short black hair."}' for _ in questions]


def test_run_squad_keeps_only_detective_yes():
    llm = _ScriptedLLM()
    queries = ["a man in red", "a woman walking"]
    cand_paths = [["cand0", "candX"], ["candX", "candX"]]  # only query0/cand0 -> "Yes"
    gate_mask = [True, True]
    out = run_squad(llm, queries, cand_paths, gate_mask)
    assert (0, 0) in out
    assert out[(0, 0)] == "A man with short black hair."
    # query0 cand1 ("candX") => "No"; query1 both "candX" => "No"
    assert (0, 1) not in out
    assert (1, 0) not in out and (1, 1) not in out


def test_run_squad_skips_gated_out_queries():
    llm = _ScriptedLLM()
    queries = ["a man in red"]
    cand_paths = [["cand0", "candX"]]
    gate_mask = [False]                          # gated out => no MLLM calls
    out = run_squad(llm, queries, cand_paths, gate_mask)
    assert out == {}
    assert llm.calls["img"] == 0 and llm.calls["txt"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_squad.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.squad'`.

- [ ] **Step 3: Implement `mllm_rerank/squad.py`**

```python
"""Single-pass Detective -> Analyst -> Writer orchestration over top-k candidates."""
import json
from PIL import Image

from mllm_rerank.prompts import (
    DETECTIVE_PROMPT, ANALYST_PROMPT, WRITER_PROMPT,
    process_checklist, parse_writer_caption,
)
from mllm_rerank.mllm import batch_infer, batch_infer_txt


def load_image(path):
    return Image.open(path).convert("RGB")


def run_squad(llm, queries, cand_img_paths, gate_mask,
              image_micro_batch=8, text_micro_batch=16, temperature=0.01):
    # ---- collect all (i, j) pairs that pass the structural gate ----
    pairs = []
    for i, passed in enumerate(gate_mask):
        if not passed:
            continue
        for j in range(len(cand_img_paths[i])):
            pairs.append((i, j))
    if not pairs:
        return {}

    # ---- Stage 1: Detective (batched) ----
    det_prompts = [DETECTIVE_PROMPT.format(cap=queries[i]) for (i, j) in pairs]
    det_images = [cand_img_paths[i][j] for (i, j) in pairs]
    det_out = batch_infer(llm, det_prompts, det_images,
                          micro_batch=image_micro_batch, t=temperature)
    survivors = [pairs[k] for k, ans in enumerate(det_out)
                 if isinstance(ans, str) and "yes" in ans.lower()]
    if not survivors:
        return {}

    # ---- Stage 2: Analyst (batched) ----
    ana_prompts = [ANALYST_PROMPT for _ in survivors]
    ana_images = [cand_img_paths[i][j] for (i, j) in survivors]
    ana_out = batch_infer(llm, ana_prompts, ana_images,
                          micro_batch=image_micro_batch, t=temperature)
    checklists = process_checklist(ana_out)   # list of list[str]

    # ---- Stage 3: Writer (batched, text-only) ----
    writer_prompts = []
    for k, (i, j) in enumerate(survivors):
        subtexts = [queries[i]] + checklists[k]
        cap_json = json.dumps(subtexts, ensure_ascii=False)
        writer_prompts.append(WRITER_PROMPT.format(cap=cap_json))
    wr_out = batch_infer_txt(llm, writer_prompts,
                             micro_batch=text_micro_batch, t=temperature)

    result = {}
    for k, (i, j) in enumerate(survivors):
        caption = parse_writer_caption(wr_out[k])
        if caption:
            result[(i, j)] = caption
    return result
```

The test's `_ScriptedLLM` works because the Detective prompt contains the word `STRICTLY` and the Analyst prompt does not; the test passes image placeholders (`"cand0"`/`"candX"`) where real code passes PIL images, and `run_squad` forwards them through unchanged. The `images[len(out)]` index in the mock tracks position within the batch.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_squad.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/squad.py tests/test_mllm_squad.py
git commit -m "feat(mllm-rerank): single-pass Detective->Analyst->Writer squad orchestration"
```

---

### Task 6: Fusion + rerank entry point + config + docs + cross-import guard

**Files:**
- Create: `mllm_rerank/fusion.py`
- Create: `mllm_rerank/rerank.py`
- Create: `mllm_rerank/config.yaml`
- Create: `mllm_rerank/run.sh`
- Create: `mllm_rerank/README.md`
- Test: `tests/test_mllm_fusion.py`
- Test: `tests/test_mllm_no_cross_import.py`

**Interfaces:**
- Consumes: `mllm_rerank.cmp_features` (`embed_texts`), `mllm_rerank.metrics` (`get_metrics`, `print_rs`), `mllm_rerank.squad` (`run_squad`, `load_image`), `mllm_rerank.mllm` (`MLLMs`).
- Produces:
  - `build_topk_and_gate(s_str: Tensor[Q,G], top_k: int, xi: float) -> (topk_idx: LongTensor[Q,k], gate_mask: list[bool])` — per query, the top-k gallery indices and whether `s_str[i].max() > xi`.
  - `fuse_scores(s_str: Tensor[Q,G], topk_idx: LongTensor[Q,k], s_sem: dict[tuple[int,int], float], lam: float) -> Tensor[Q,G]` — `S_final`. For each `(i, local_j)` in `s_sem` (local_j indexes into `topk_idx[i]`), set `S_final[i, topk_idx[i, local_j]] = lam*s_str + (1-lam)*s_sem_norm`; all other positions keep `s_str`. `s_sem` values are min-max normalized across the dict first (skip if <2 entries; clamp raw to [0,1]).
  - `rerank.py` `__main__`: orchestrates Stage-1 (or cache load) → top-k/gate → `run_squad` → `embed_texts` for `S_sem` → `fuse_scores` → `print_rs` (base vs rerank) → save outputs.

- [ ] **Step 1: Write the failing test**

`tests/test_mllm_fusion.py`:

```python
import torch
from mllm_rerank.fusion import build_topk_and_gate, fuse_scores


def test_build_topk_and_gate():
    s = torch.tensor([[0.9, 0.1, 0.5], [0.05, 0.02, 0.03]])
    topk_idx, gate = build_topk_and_gate(s, top_k=2, xi=0.1)
    assert topk_idx.shape == (2, 2)
    assert topk_idx[0].tolist() == [0, 2]      # 0.9, 0.5
    assert gate == [True, False]               # row1 max 0.9 > 0.1; row2 max 0.05 <= 0.1


def test_fuse_only_touches_processed_positions():
    s = torch.tensor([[0.8, 0.2, 0.5, 0.1]])
    topk_idx = torch.tensor([[0, 2]])          # candidates at gallery idx 0 and 2
    # semantic score only for local_j=1 (gallery idx 2); two entries to enable min-max
    s_sem = {(0, 0): 0.0, (0, 1): 1.0}
    out = fuse_scores(s, topk_idx, s_sem, lam=0.4)
    # local 0 -> gallery 0: 0.4*0.8 + 0.6*0.0 = 0.32
    assert abs(out[0, 0].item() - 0.32) < 1e-5
    # local 1 -> gallery 2: 0.4*0.5 + 0.6*1.0 = 0.80
    assert abs(out[0, 2].item() - 0.80) < 1e-5
    # untouched positions keep s_str
    assert out[0, 1].item() == 0.2
    assert out[0, 3].item() == 0.1


def test_fuse_promotes_high_semantic_candidate_in_ranking():
    s = torch.tensor([[0.8, 0.79]])            # idx0 slightly ahead structurally
    topk_idx = torch.tensor([[0, 1]])
    s_sem = {(0, 0): 0.0, (0, 1): 1.0}         # idx1 strongly favored semantically
    out = fuse_scores(s, topk_idx, s_sem, lam=0.4)
    assert out[0, 1] > out[0, 0]               # idx1 now ranks first
```

`tests/test_mllm_no_cross_import.py`:

```python
import glob


def test_no_module_imports_across_open_sources():
    for path in glob.glob("mllm_rerank/*.py"):
        with open(path) as f:
            src = f.read()
        assert "open-sources" not in src, f"{path} references open-sources"
        assert "sys.path" not in src, f"{path} mutates sys.path"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_fusion.py tests/test_mllm_no_cross_import.py -q`
Expected: `test_mllm_fusion.py` FAILs with `ModuleNotFoundError: No module named 'mllm_rerank.fusion'`. (`test_mllm_no_cross_import.py` may already pass for existing files — that is fine; it must pass after this task.)

- [ ] **Step 3: Implement `mllm_rerank/fusion.py`**

```python
"""Top-k gating + Eq.4 fusion of structural (S_str) and semantic (S_sem) scores."""
import torch


def build_topk_and_gate(s_str, top_k, xi):
    k = min(top_k, s_str.size(1))
    topk_idx = s_str.topk(k=k, dim=1).indices
    row_max = s_str.max(dim=1).values
    gate_mask = (row_max > xi).tolist()
    return topk_idx, gate_mask


def fuse_scores(s_str, topk_idx, s_sem, lam):
    s_final = s_str.clone()
    if not s_sem:
        return s_final
    # min-max normalize semantic scores across processed (i,j) entries
    vals = list(s_sem.values())
    if len(vals) >= 2:
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        norm = {kk: (vv - lo) / denom for kk, vv in s_sem.items()}
    else:
        norm = {kk: max(0.0, min(1.0, vv)) for kk, vv in s_sem.items()}
    for (i, local_j), sem in norm.items():
        g = int(topk_idx[i, local_j].item())
        s_final[i, g] = lam * float(s_str[i, g]) + (1 - lam) * float(sem)
    return s_final
```

- [ ] **Step 4: Run fusion + guard tests to verify they pass**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_fusion.py tests/test_mllm_no_cross_import.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Implement `mllm_rerank/rerank.py`** (entry point — wires everything; no new unit test, validated by the deferred smoke + `ast.parse`)

```python
"""Entry: CMP Stage-1 -> top-k gate -> Detective Squad -> semantic fusion -> metrics."""
import os
import gc
import json
import time
import argparse
import torch
from ruamel.yaml import YAML

from mllm_rerank.cmp_features import (
    load_cmp_components, extract_cmp_features, compute_cmp_itm_scores, embed_texts,
)
from mllm_rerank.metrics import get_metrics, print_rs
from mllm_rerank.fusion import build_topk_and_gate, fuse_scores
from mllm_rerank.squad import run_squad, load_image
from mllm_rerank.mllm import MLLMs

yaml = YAML(typ="safe")


def _simple_logger(out_dir):
    import logging
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("mllm_rerank")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(out_dir, "rerank.log"))
        fmt = logging.Formatter("%(asctime)s %(message)s")
        sh.setFormatter(fmt); fh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.load(open(args.config, "r"))
    out_dir = cfg["out_dir"]
    logger = _simple_logger(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Stage 1: CMP features + ITM structural score (cache) ----
    cache = os.path.join(out_dir, "cmp_features.pt")
    if os.path.exists(cache):
        logger.info(f"Loading cached CMP features: {cache}")
        c = torch.load(cache)
        s_str = c["s_str"]; img_paths = c["img_paths"]; texts = c["texts"]
        qids = c["q_pids"]; pids = c["g_pids"]
        text_embeds_meta = None
        cmp_config = c["cmp_config"]
    else:
        from dataset import create_dataset, create_loader
        cmp_config, tokenizer, model = load_cmp_components(
            cfg["cmp_config"], cfg["cmp_checkpoint"], device)
        _, ds = create_dataset(cmp_config, evaluate=True)
        loader = create_loader([ds], [None], batch_size=[cmp_config["batch_size_test"]],
                               num_workers=[4], is_trains=[False], collate_fns=[None])[0]
        feats = extract_cmp_features(model, loader, tokenizer, device, cmp_config)
        s_str = compute_cmp_itm_scores(model, feats, device, cmp_config)
        img_paths = feats["img_paths"]; texts = feats["texts"]
        qids = torch.tensor(ds.q_pids); pids = torch.tensor(ds.g_pids)
        torch.save({"s_str": s_str.cpu(), "img_paths": img_paths, "texts": texts,
                    "q_pids": qids, "g_pids": pids, "cmp_config": cmp_config}, cache)
        del model, tokenizer, loader, ds, feats
        torch.cuda.empty_cache(); gc.collect(); time.sleep(5)

    qids = torch.as_tensor(qids); pids = torch.as_tensor(pids)
    base_row = get_metrics(s_str.cpu(), qids, pids, "CMP-Base", False)
    logger.info(f"Base CMP: R1={base_row[1]:.2f} R5={base_row[2]:.2f} "
                f"R10={base_row[3]:.2f} mAP={base_row[4]:.2f}")

    # ---- top-k + gate ----
    topk_idx, gate_mask = build_topk_and_gate(s_str, cfg["top_k"], cfg["xi"])
    cand_img_paths = [[img_paths[int(topk_idx[i, j])] for j in range(topk_idx.size(1))]
                      for i in range(topk_idx.size(0))]
    # load PIL images lazily inside run_squad's batches: pass paths, convert there
    cand_imgs = [[load_image(p) for p in row] if gate_mask[i] else []
                 for i, row in enumerate(cand_img_paths)]

    # ---- Stage 2: load vLLM, run squad ----
    llm = MLLMs(cfg["model_dir"],
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.7),
                max_model_len=cfg.get("max_model_len", 1536))
    new_caps = run_squad(
        llm, list(texts), cand_imgs, gate_mask,
        image_micro_batch=cfg.get("image_micro_batch", 8),
        text_micro_batch=cfg.get("text_micro_batch", 16),
        temperature=cfg.get("temperature", 0.01),
    )
    logger.info(f"Squad produced {len(new_caps)} new captions")
    del llm; torch.cuda.empty_cache(); gc.collect(); time.sleep(2)

    # ---- Stage 3: semantic cosine via frozen CMP BERT tower ----
    _, tok2, model2 = load_cmp_components(cfg["cmp_config"], cfg["cmp_checkpoint"], device)
    keys = list(new_caps.keys())
    s_sem = {}
    if keys:
        new_texts = [new_caps[k] for k in keys]
        query_texts = [texts[i] for (i, j) in keys]
        e_new = embed_texts(model2, tok2, new_texts, device, cfg)
        e_q = embed_texts(model2, tok2, query_texts, device, cfg)
        cos = (e_q * e_new).sum(dim=-1)               # both already L2-normalized
        for idx, (i, j) in enumerate(keys):
            s_sem[(i, j)] = float(cos[idx].item())

    # ---- fuse + metrics ----
    s_final = fuse_scores(s_str, topk_idx, s_sem, cfg["lambda"])
    print_rs({"sims_base": s_str, "sims_rerank": s_final}, qids, pids, logger)

    torch.save({"s_final": s_final.cpu(), "new_captions": {f"{i}_{j}": v
                for (i, j), v in new_caps.items()}},
               os.path.join(out_dir, "rerank_result.pt"))
    with open(os.path.join(out_dir, "new_captions.json"), "w", encoding="utf-8") as f:
        json.dump({f"{i}_{j}": v for (i, j), v in new_caps.items()}, f,
                  ensure_ascii=False, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify `rerank.py` parses**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -c "import ast; ast.parse(open('mllm_rerank/rerank.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Create `mllm_rerank/config.yaml`**

```yaml
# --- Stage-1 CMP (repo Search model) ---
cmp_config: 'configs/PAB.yaml'              # EDIT: path to the trained CMP stage-1 yaml
cmp_checkpoint: 'output/cmp/best.pth'        # EDIT: path to the trained CMP checkpoint

# --- MLLM (Qwen3-VL-8B-Instruct, zero-shot) ---
model_dir: 'checkpoint/Qwen3-VL-8B-Instruct' # EDIT: local path to the MLLM weights
gpu_memory_utilization: 0.7
max_model_len: 1536
image_micro_batch: 8
text_micro_batch: 16
temperature: 0.01

# --- Rerank ---
top_k: 10
xi: 0.1
lambda: 0.4
max_tokens_text: 56

out_dir: 'output/mllm_rerank'
```

- [ ] **Step 8: Create `mllm_rerank/run.sh`**

```bash
#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p output/mllm_rerank logs
python -m mllm_rerank.rerank --config mllm_rerank/config.yaml \
    2>&1 | tee logs/mllm_rerank_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 9: Create `mllm_rerank/README.md`**

````markdown
# MLLM Agent Rerank (SSDC Stage-2)

Reranks the CMP Stage-1 retrieval with a single-pass **Detective → Analyst →
Writer** agent chain (Qwen3-VL-8B-Instruct, zero-shot, served via vLLM),
following the SSDC paper's **semantic-cosine** fusion.

## Pipeline
1. CMP Stage-1 → ITM structural score `S_str` (cached to `out_dir/cmp_features.pt`).
2. Per query: top-k candidates by `S_str`, gated on `xi`.
3. Detective (Yes/No) → Analyst (15-item checklist) → Writer (new caption `T_new`).
4. `S_sem = cos(E_txt(T), E_txt(T_new))` via the frozen CMP BERT text tower.
5. Fuse `S_final = λ·S_str + (1-λ)·S_sem` on processed positions; rerank.

## Run
```bash
# edit mllm_rerank/config.yaml paths first
bash mllm_rerank/run.sh
```
Requires: `vllm`, `qwen_vl_utils`, a trained CMP checkpoint + yaml, PAB test data,
local Qwen3-VL-8B-Instruct weights, `bert-base-uncased` for the CMP text tower.

## Note: paper vs released SSDC code
The released SSDC code reranks by **hard-boosting** the MLLM-verified candidate's
similarity to 1.0. This module instead follows the **paper** (Eq. 2-4): it scores
each candidate by the cosine between the original query and the Writer's
**generated caption**. All SSDC logic here is **ported (copied)** into this
package — nothing is imported from `open-sources/SSDC`.
````

- [ ] **Step 10: Run the full module test suite + cross-import guard**

Run: `cd ~/Code/Project/Sim2Real-Track4 && python -m pytest tests/test_mllm_prompts.py tests/test_mllm_metrics.py tests/test_mllm_cmp_features.py tests/test_mllm_wrapper.py tests/test_mllm_squad.py tests/test_mllm_fusion.py tests/test_mllm_no_cross_import.py -q`
Expected: PASS (all tests across the 7 files green).

- [ ] **Step 11: Commit**

```bash
git add mllm_rerank/fusion.py mllm_rerank/rerank.py mllm_rerank/config.yaml \
        mllm_rerank/run.sh mllm_rerank/README.md \
        tests/test_mllm_fusion.py tests/test_mllm_no_cross_import.py
git commit -m "feat(mllm-rerank): Eq.4 fusion + rerank entry point + config/run/docs"
```

---

## Deferred Smoke Test (needs prereqs)

When Qwen3-VL-8B-Instruct weights, `vllm`, a trained CMP checkpoint, and PAB test
data are all available on a GPU box:

```bash
# 1. edit mllm_rerank/config.yaml (cmp_config, cmp_checkpoint, model_dir)
bash mllm_rerank/run.sh
```

Confirm: the log prints the **Base CMP** row, then a PrettyTable with `sims_base`
vs `sims_rerank`; `output/mllm_rerank/new_captions.json` is non-empty; the
`sims_rerank` mAP/R@1 is >= base (or document the delta). If vLLM OOMs alongside
CMP, the cache path already releases the CMP model before loading vLLM — rerun
once `cmp_features.pt` exists so Stage-1 is skipped.
