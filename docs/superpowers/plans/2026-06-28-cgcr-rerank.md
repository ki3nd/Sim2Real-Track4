# CGCR — Claim-Grounded Comparative Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second MLLM rerank entry point `mllm_rerank/cgcr.py` that reranks CMP Stage-1 candidates by weighted signed visual-entailment over decomposed query claims, with self-consistency, a targeted Critic, ambiguity gating, adaptive recall recovery, and a listwise Adjudicator tie-break — all zero-shot.

**Architecture:** New files under the existing `mllm_rerank/` package (`cgcr_prompts.py`, `claims.py`, `cgcr_gate.py`, `cgcr_squad.py`, `cgcr.py`, `cgcr_config.yaml`). It reuses the merged `mllm_rerank.mllm` (vLLM `MLLMs` + batch helpers), `mllm_rerank.metrics`, `mllm_rerank.cmp_features` (shared Stage-1 cache), and `mllm_rerank.fusion.fuse_scores`. CGCR does NOT modify or replace the existing `rerank.py`.

**Tech Stack:** Python, PyTorch, vLLM (deferred import), `ruamel.yaml`, pytest. All unit tests run on CPU with mock LLM objects; the GPU/vLLM smoke is deferred.

## Global Constraints

- **No cross-repo import:** no file under `mllm_rerank/` may contain the string `open-sources` or mutate `sys.path`. (Test-enforced; extend the existing guard in Task 5.)
- **Zero-shot MLLM only:** no fine-tuning / LoRA / hard-negative mining / training of any model.
- **Score signal is weighted signed visual entailment** (`+conf` ENTAILED / `0` NEUTRAL / `-conf` CONTRADICTED), NOT text-text cosine and NOT hard-boost.
- **Claim weights are deterministic:** the MLLM assigns a category + extracts claim text; the weight is looked up from a fixed `category_weights` table (config), never trusted from the MLLM.
- **Do not modify** `mllm_rerank/rerank.py`, `prompts.py`, `mllm.py`, `metrics.py`, `cmp_features.py`, `fusion.py` — reuse them by import only.
- **Reuse, don't duplicate:** `MLLMs`, `batch_infer`, `batch_infer_txt` (from `mllm_rerank.mllm`), `fuse_scores` (from `mllm_rerank.fusion`), `get_metrics`/`print_rs` (from `mllm_rerank.metrics`), `cmp_features` Stage-1 helpers.
- **Tests** are plain pytest functions under `tests/` (no classes/fixtures), matching the existing `tests/test_mllm_*.py` style. Run with the project venv: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest <files> -q` (system python lacks torch).
- **Defaults** (from spec §7): `n_samples=3`, `gate_margin=0.05`, `k0=5`, `k_step=5`, `k_max=20`, `hit_theta=0.6`, `max_rounds=2`, `lambda=0.4`, `tie_margin=0.03`, `critic_conf_floor=0.5`.

---

### Task 1: CGCR prompts + parsers

**Files:**
- Create: `mllm_rerank/cgcr_prompts.py`
- Test: `tests/test_cgcr_prompts.py`

**Interfaces:**
- Produces:
  - `DECOMPOSER_PROMPT: str` (field `{query}`), `VERIFIER_PROMPT: str` (field `{claims}`), `CRITIC_PROMPT: str` (field `{claims}`), `ADJUDICATOR_PROMPT: str` (fields `{query}`, `{n}`).
  - `parse_claims(raw: str) -> list[dict]` — each dict `{"id": int, "category": str, "text": str}`. Sequential ids from 1 if missing/duplicated. `[]` on failure.
  - `parse_verdicts(raw: str) -> list[dict]` — each `{"id": int, "label": str, "conf": float, "evidence": str}`. `label` upper-cased and restricted to `ENTAILED`/`NEUTRAL`/`CONTRADICTED` (anything else → `NEUTRAL`); `conf` clamped to `[0,1]` (default `0.0`). `[]` on failure.
  - `parse_adjudication(raw: str, n: int) -> list[int]` — a permutation of `0..n-1`; identity `list(range(n))` on failure or malformed.

- [ ] **Step 1: Write the failing test**

`tests/test_cgcr_prompts.py`:

```python
from mllm_rerank.cgcr_prompts import (
    DECOMPOSER_PROMPT, VERIFIER_PROMPT, CRITIC_PROMPT, ADJUDICATOR_PROMPT,
    parse_claims, parse_verdicts, parse_adjudication,
)


def test_prompt_fields():
    assert "{query}" in DECOMPOSER_PROMPT
    assert "{claims}" in VERIFIER_PROMPT
    assert "{claims}" in CRITIC_PROMPT
    assert "{query}" in ADJUDICATOR_PROMPT and "{n}" in ADJUDICATOR_PROMPT


def test_parse_claims_valid():
    raw = '{"claims":[{"id":1,"category":"action","text":"falling down"},{"id":2,"category":"upper","text":"red shirt"}]}'
    out = parse_claims(raw)
    assert len(out) == 2
    assert out[0] == {"id": 1, "category": "action", "text": "falling down"}
    assert out[1]["category"] == "upper"


def test_parse_claims_assigns_sequential_ids_when_missing():
    raw = '{"claims":[{"category":"gender","text":"male"},{"category":"action","text":"running"}]}'
    out = parse_claims(raw)
    assert [c["id"] for c in out] == [1, 2]


def test_parse_claims_bad_json_returns_empty():
    assert parse_claims("not json at all") == []


def test_parse_verdicts_normalizes_label_and_clamps_conf():
    raw = '{"verdicts":[{"id":1,"label":"entailed","conf":1.4,"evidence":"x"},{"id":2,"label":"weird","conf":-0.2,"evidence":"y"}]}'
    out = parse_verdicts(raw)
    assert out[0]["label"] == "ENTAILED" and out[0]["conf"] == 1.0
    assert out[1]["label"] == "NEUTRAL" and out[1]["conf"] == 0.0


def test_parse_verdicts_embedded_json():
    raw = 'Here:\n{"verdicts":[{"id":3,"label":"CONTRADICTED","conf":0.7,"evidence":"push-ups"}]} done'
    out = parse_verdicts(raw)
    assert len(out) == 1 and out[0]["id"] == 3 and out[0]["label"] == "CONTRADICTED"


def test_parse_adjudication_valid_permutation():
    assert parse_adjudication('{"ranking":[2,0,1]}', 3) == [2, 0, 1]


def test_parse_adjudication_fallback_identity():
    assert parse_adjudication("garbage", 3) == [0, 1, 2]
    # incomplete / out-of-range permutation -> identity
    assert parse_adjudication('{"ranking":[0,0,5]}', 3) == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_prompts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.cgcr_prompts'`.

- [ ] **Step 3: Implement `mllm_rerank/cgcr_prompts.py`**

```python
"""CGCR agent prompts (Decomposer / Verifier / Critic / Adjudicator) + JSON parsers."""
import json
import re

DECOMPOSER_PROMPT = """You are decomposing a person-search query into atomic, independently checkable claims.
For each claim, assign a category from this set: gender, hair, upper, lower, accessory, action, anomaly, background.
Use "action" or "anomaly" for what the person is DOING (e.g., walking, falling, fighting).

Query: {query}

Return STRICTLY a JSON object: {{"claims":[{{"id":1,"category":"<cat>","text":"<short claim>"}}, ...]}}.
Keep each claim text short (a few words). Do not add commentary."""

VERIFIER_PROMPT = """You are verifying whether each claim is true of the PRIMARY person in the image.
For each claim, answer:
- "ENTAILED" if the image clearly supports it,
- "CONTRADICTED" if the image clearly shows the opposite,
- "NEUTRAL" if it cannot be determined from the image.
Give a confidence in [0,1] and one short evidence phrase.

Claims (JSON): {claims}

Return STRICTLY a JSON object: {{"verdicts":[{{"id":1,"label":"ENTAILED|NEUTRAL|CONTRADICTED","conf":0.0,"evidence":"<short>"}}, ...]}}."""

CRITIC_PROMPT = """Look again at the image and re-judge ONLY the following suspect claims.
Be strict: confirm a claim only if visually evident; otherwise mark NEUTRAL or CONTRADICTED.

Suspect claims (JSON): {claims}

Return STRICTLY a JSON object: {{"verdicts":[{{"id":1,"label":"ENTAILED|NEUTRAL|CONTRADICTED","conf":0.0,"evidence":"<short>"}}, ...]}}."""

ADJUDICATOR_PROMPT = """You are given a query and {n} candidate images (in order, index 0..{n}-1).
Rank the images from best to worst match to the query, focusing on the described ACTION/behavior first, then appearance.

Query: {query}

Return STRICTLY a JSON object: {{"ranking":[<best index>, ..., <worst index>]}} listing each index 0..{n}-1 exactly once."""


def _extract_json(text):
    """Return the first JSON object parseable from text, else None."""
    if text is None:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def parse_claims(raw):
    obj = _extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        return []
    out = []
    for i, c in enumerate(obj["claims"], start=1):
        if not isinstance(c, dict):
            continue
        out.append({
            "id": i,
            "category": str(c.get("category", "")).strip().lower(),
            "text": str(c.get("text", "")).strip(),
        })
    return out


_LABELS = {"ENTAILED", "NEUTRAL", "CONTRADICTED"}


def parse_verdicts(raw):
    obj = _extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("verdicts"), list):
        return []
    out = []
    for v in obj["verdicts"]:
        if not isinstance(v, dict) or "id" not in v:
            continue
        label = str(v.get("label", "")).strip().upper()
        if label not in _LABELS:
            label = "NEUTRAL"
        try:
            conf = float(v.get("conf", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append({
            "id": int(v["id"]),
            "label": label,
            "conf": conf,
            "evidence": str(v.get("evidence", "")).strip(),
        })
    return out


def parse_adjudication(raw, n):
    identity = list(range(n))
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return identity
    rank = obj.get("ranking")
    if not isinstance(rank, list) or len(rank) != n:
        return identity
    try:
        rank = [int(x) for x in rank]
    except Exception:
        return identity
    if sorted(rank) != identity:   # must be a permutation of 0..n-1
        return identity
    return rank
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_prompts.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/cgcr_prompts.py tests/test_cgcr_prompts.py
git commit -m "feat(cgcr): agent prompts + JSON parsers (claims/verdicts/adjudication)"
```

---

### Task 2: Claim weighting + self-consistency aggregation + grounded score

**Files:**
- Create: `mllm_rerank/claims.py`
- Test: `tests/test_cgcr_claims.py`

**Interfaces:**
- Consumes: nothing (pure logic; verdict/claim dicts have the shape produced by Task 1).
- Produces:
  - `category_weight(category: str, table: dict, default: float) -> float` — `table.get(category.lower(), default)`.
  - `aggregate_self_consistency(samples: list[list[dict]]) -> list[dict]` — `samples` is N verdict-lists (Task-1 shape). Group by `id`; per id pick the **majority label**, `conf` = mean conf of the verdicts that have the majority label; on a label-count tie prefer `NEUTRAL`. Returns one verdict dict per id `{"id", "label", "conf"}` sorted by id.
  - `score_ground(claims: list[dict], verdicts: list[dict], table: dict, default: float) -> float` — map verdicts by id; for each claim `w = category_weight(claim['category'], table, default)`, `v = +conf if ENTAILED, 0 if NEUTRAL/missing, -conf if CONTRADICTED`; return `sum(w*v)/sum(w)` (0.0 if `sum(w)==0`).

- [ ] **Step 1: Write the failing test**

`tests/test_cgcr_claims.py`:

```python
from mllm_rerank.claims import category_weight, aggregate_self_consistency, score_ground

TABLE = {"action": 1.5, "upper": 1.0, "gender": 0.8, "background": 0.3}
DEFAULT = 0.7


def test_category_weight_lookup_and_default():
    assert category_weight("action", TABLE, DEFAULT) == 1.5
    assert category_weight("ACTION", TABLE, DEFAULT) == 1.5
    assert category_weight("unknown", TABLE, DEFAULT) == DEFAULT


def test_aggregate_majority_label_and_mean_conf():
    samples = [
        [{"id": 1, "label": "ENTAILED", "conf": 0.8}],
        [{"id": 1, "label": "ENTAILED", "conf": 0.6}],
        [{"id": 1, "label": "CONTRADICTED", "conf": 0.9}],
    ]
    out = aggregate_self_consistency(samples)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["label"] == "ENTAILED"
    assert abs(out[0]["conf"] - 0.7) < 1e-6   # mean of the two ENTAILED confs


def test_aggregate_label_tie_prefers_neutral():
    samples = [
        [{"id": 5, "label": "ENTAILED", "conf": 0.9}],
        [{"id": 5, "label": "CONTRADICTED", "conf": 0.9}],
    ]
    out = aggregate_self_consistency(samples)
    assert out[0]["label"] == "NEUTRAL"


def test_score_ground_signs_and_weights():
    claims = [
        {"id": 1, "category": "action", "text": "falling"},
        {"id": 2, "category": "upper", "text": "red shirt"},
        {"id": 3, "category": "gender", "text": "male"},
    ]
    verdicts = [
        {"id": 1, "label": "CONTRADICTED", "conf": 1.0},  # -1.5 weight contribution
        {"id": 2, "label": "ENTAILED", "conf": 1.0},      # +1.0
        {"id": 3, "label": "NEUTRAL", "conf": 0.5},       # 0
    ]
    s = score_ground(claims, verdicts, TABLE, DEFAULT)
    # (1.5*-1 + 1.0*1 + 0.8*0) / (1.5+1.0+0.8) = -0.5/3.3
    assert abs(s - (-0.5 / 3.3)) < 1e-6


def test_score_ground_all_neutral_is_zero():
    claims = [{"id": 1, "category": "action", "text": "x"}]
    verdicts = [{"id": 1, "label": "NEUTRAL", "conf": 0.9}]
    assert score_ground(claims, verdicts, TABLE, DEFAULT) == 0.0


def test_score_ground_missing_verdict_treated_neutral():
    claims = [{"id": 1, "category": "action", "text": "x"}]
    assert score_ground(claims, [], TABLE, DEFAULT) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_claims.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.claims'`.

- [ ] **Step 3: Implement `mllm_rerank/claims.py`**

```python
"""Claim weighting, self-consistency aggregation, and the grounded entailment score."""
from collections import defaultdict


def category_weight(category, table, default):
    return table.get(str(category).strip().lower(), default)


def aggregate_self_consistency(samples):
    """samples: list of verdict-lists (each verdict {id,label,conf}). Majority label per id;
    conf = mean conf of majority-label verdicts; label-count tie -> NEUTRAL."""
    by_id = defaultdict(list)
    for sample in samples:
        for v in sample:
            by_id[v["id"]].append(v)

    out = []
    for vid in sorted(by_id.keys()):
        verds = by_id[vid]
        counts = defaultdict(int)
        for v in verds:
            counts[v["label"]] += 1
        top = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == top]
        if len(winners) > 1:
            label = "NEUTRAL"
        else:
            label = winners[0]
        confs = [v["conf"] for v in verds if v["label"] == label]
        conf = sum(confs) / len(confs) if confs else 0.0
        out.append({"id": vid, "label": label, "conf": conf})
    return out


def score_ground(claims, verdicts, table, default):
    vmap = {v["id"]: v for v in verdicts}
    num = 0.0
    den = 0.0
    for c in claims:
        w = category_weight(c.get("category", ""), table, default)
        den += w
        v = vmap.get(c["id"])
        if v is None:
            continue
        if v["label"] == "ENTAILED":
            num += w * v["conf"]
        elif v["label"] == "CONTRADICTED":
            num += w * (-v["conf"])
        # NEUTRAL contributes 0
    if den == 0.0:
        return 0.0
    return num / den
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_claims.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/claims.py tests/test_cgcr_claims.py
git commit -m "feat(cgcr): claim weighting + self-consistency aggregation + grounded score"
```

---

### Task 3: Ambiguity gate

**Files:**
- Create: `mllm_rerank/cgcr_gate.py`
- Test: `tests/test_cgcr_gate.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ambiguity_gate(s_str, gate_margin: float) -> (gate_mask: list[bool], margins: list[float])` — `s_str` is a `Tensor[Q,G]`. For each query row, `margin = top1 - top2` of the row's values; `gate_mask[i] = margin < gate_margin` (activate MLLM only when the top-1/top-2 gap is small, i.e. ambiguous). For a single-column gallery, margin is the top value (no second). Returns the bool list and the float margins (for logging).

- [ ] **Step 1: Write the failing test**

`tests/test_cgcr_gate.py`:

```python
import torch
from mllm_rerank.cgcr_gate import ambiguity_gate


def test_peaked_row_gated_out_flat_row_activated():
    s = torch.tensor([
        [0.9, 0.1, 0.05],   # margin 0.8 -> not ambiguous -> skip (False)
        [0.50, 0.49, 0.2],  # margin 0.01 -> ambiguous   -> activate (True)
    ])
    mask, margins = ambiguity_gate(s, gate_margin=0.05)
    assert mask == [False, True]
    assert abs(margins[0] - 0.8) < 1e-6
    assert abs(margins[1] - 0.01) < 1e-6


def test_boundary_is_strict_less_than():
    s = torch.tensor([[0.5, 0.45]])   # margin exactly 0.05
    mask, _ = ambiguity_gate(s, gate_margin=0.05)
    assert mask == [False]            # 0.05 < 0.05 is False -> skip
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_gate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.cgcr_gate'`.

- [ ] **Step 3: Implement `mllm_rerank/cgcr_gate.py`**

```python
"""Ambiguity gate: activate the MLLM rerank only when Stage-1 is undecided (small top1-top2 margin)."""
import torch


def ambiguity_gate(s_str, gate_margin):
    mask, margins = [], []
    for i in range(s_str.size(0)):
        row = s_str[i]
        k = min(2, row.size(0))
        top = torch.topk(row, k=k).values
        if k >= 2:
            margin = float(top[0] - top[1])
        else:
            margin = float(top[0])
        margins.append(margin)
        mask.append(margin < gate_margin)
    return mask, margins
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/cgcr_gate.py tests/test_cgcr_gate.py
git commit -m "feat(cgcr): ambiguity gate (top1-top2 margin)"
```

---

### Task 4: Squad orchestration (decompose / verify / critic + recovery control)

**Files:**
- Create: `mllm_rerank/cgcr_squad.py`
- Test: `tests/test_cgcr_squad.py`

**Interfaces:**
- Consumes: `mllm_rerank.cgcr_prompts` (`DECOMPOSER_PROMPT`, `VERIFIER_PROMPT`, `CRITIC_PROMPT`, `parse_claims`, `parse_verdicts`); `mllm_rerank.mllm` (`batch_infer`, `batch_infer_txt`); `mllm_rerank.claims` (`aggregate_self_consistency`).
- Produces:
  - `decompose(llm, queries: list[str], text_micro_batch=16) -> list[list[dict]]` — one Decomposer call per query (text-only, batched); returns per-query claim lists (Task-1 `parse_claims` shape).
  - `verify_pairs(llm, work_items, n_samples=3, temperature=0.6, image_micro_batch=8) -> list[list[dict]]` — `work_items` is a list of `(claims, image)`. For each item, run the Verifier `n_samples` times and aggregate via `aggregate_self_consistency`. Returns one aggregated verdict-list per item (same order). Empty `work_items` → `[]`.
  - `critic_pass(llm, suspect_claims, image, image_micro_batch=8) -> list[dict]` — one Critic call for a single image over the suspect claim list; returns parsed verdicts (may be empty). (Batched callers loop over images.)
  - `should_continue_recovery(best_ground: float, current_k: int, round_idx: int, hit_theta: float, k_step: int, k_max: int, max_rounds: int) -> int | None` — returns the next `k` to verify (`current_k + k_step`) to continue the recall-recovery loop, or `None` to stop. Stop when `best_ground >= hit_theta`, or `round_idx + 1 >= max_rounds`, or `current_k >= k_max`. Never returns a `k` above `k_max` (clamps).
- **Adjudicator NOT implemented here:** `ADJUDICATOR_PROMPT` + `parse_adjudication` (Task 1, tested) are the ready scaffolding, but the `adjudicate()` function is deferred until a multi-image `MLLMs` method exists (the single-image wrapper cannot compare candidates). Do not add an `adjudicate()` that only sees one image.

- [ ] **Step 1: Write the failing test**

`tests/test_cgcr_squad.py`:

```python
from mllm_rerank.cgcr_squad import (
    decompose, verify_pairs, should_continue_recovery,
)


class _LLM:
    """Decomposer (text) returns one claim; Verifier (image) returns ENTAILED for image=='match' else CONTRADICTED."""
    def generate_response_text(self, questions, t=0.01):
        return ['{"claims":[{"id":1,"category":"action","text":"falling"}]}' for _ in questions]

    def generate_response_multi_images(self, questions, images, t=0.01):
        out = []
        for img in images:
            if img == "match":
                out.append('{"verdicts":[{"id":1,"label":"ENTAILED","conf":0.9,"evidence":"e"}]}')
            else:
                out.append('{"verdicts":[{"id":1,"label":"CONTRADICTED","conf":0.8,"evidence":"e"}]}')
        return out


def test_decompose_returns_claims_per_query():
    out = decompose(_LLM(), ["q1", "q2"])
    assert len(out) == 2
    assert out[0][0]["category"] == "action"


def test_verify_pairs_aggregates_self_consistency():
    claims = [{"id": 1, "category": "action", "text": "falling"}]
    work = [(claims, "match"), (claims, "other")]
    out = verify_pairs(_LLM(), work, n_samples=3)
    assert len(out) == 2
    assert out[0][0]["label"] == "ENTAILED"     # 3/3 ENTAILED
    assert out[1][0]["label"] == "CONTRADICTED"  # 3/3 CONTRADICTED


def test_verify_pairs_empty():
    assert verify_pairs(_LLM(), [], n_samples=3) == []


def test_should_continue_stops_on_hit():
    assert should_continue_recovery(0.7, 5, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) is None


def test_should_continue_deepens_when_below_hit():
    assert should_continue_recovery(0.3, 5, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) == 10


def test_should_continue_stops_at_kmax():
    assert should_continue_recovery(0.3, 20, 0, hit_theta=0.6, k_step=5, k_max=20, max_rounds=5) is None


def test_should_continue_stops_at_max_rounds():
    assert should_continue_recovery(0.3, 5, 1, hit_theta=0.6, k_step=5, k_max=20, max_rounds=2) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_squad.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.cgcr_squad'`.

- [ ] **Step 3: Implement `mllm_rerank/cgcr_squad.py`**

```python
"""CGCR squad: decompose -> verify (self-consistency) -> critic, plus recall-recovery control.

The Adjudicator is deferred (see plan): the single-image MLLMs wrapper cannot compare
candidates, so adjudicate() is not implemented here. ADJUDICATOR_PROMPT / parse_adjudication
remain in cgcr_prompts as ready scaffolding.
"""
from mllm_rerank.cgcr_prompts import (
    DECOMPOSER_PROMPT, VERIFIER_PROMPT, CRITIC_PROMPT,
    parse_claims, parse_verdicts,
)
from mllm_rerank.claims import aggregate_self_consistency
from mllm_rerank.mllm import batch_infer, batch_infer_txt
import json


def decompose(llm, queries, text_micro_batch=16):
    prompts = [DECOMPOSER_PROMPT.format(query=q) for q in queries]
    raw = batch_infer_txt(llm, prompts, micro_batch=text_micro_batch, t=0.01)
    return [parse_claims(r) for r in raw]


def verify_pairs(llm, work_items, n_samples=3, temperature=0.6, image_micro_batch=8):
    if not work_items:
        return []
    # Build n_samples repeats of every item, run all in one batched call, then regroup.
    flat_prompts, flat_images = [], []
    for claims, image in work_items:
        cap = json.dumps([{"id": c["id"], "category": c["category"], "text": c["text"]} for c in claims],
                         ensure_ascii=False)
        for _ in range(n_samples):
            flat_prompts.append(VERIFIER_PROMPT.format(claims=cap))
            flat_images.append(image)
    flat_raw = batch_infer(llm, flat_prompts, flat_images, micro_batch=image_micro_batch, t=temperature)

    out = []
    for idx in range(len(work_items)):
        start = idx * n_samples
        samples = [parse_verdicts(flat_raw[start + s]) for s in range(n_samples)]
        out.append(aggregate_self_consistency(samples))
    return out


def critic_pass(llm, suspect_claims, image, image_micro_batch=8):
    if not suspect_claims:
        return []
    cap = json.dumps([{"id": c["id"], "category": c["category"], "text": c["text"]} for c in suspect_claims],
                     ensure_ascii=False)
    raw = batch_infer(llm, [CRITIC_PROMPT.format(claims=cap)], [image],
                      micro_batch=image_micro_batch, t=0.01)
    return parse_verdicts(raw[0]) if raw else []


def should_continue_recovery(best_ground, current_k, round_idx, hit_theta, k_step, k_max, max_rounds):
    if best_ground >= hit_theta:
        return None
    if round_idx + 1 >= max_rounds:
        return None
    if current_k >= k_max:
        return None
    return min(current_k + k_step, k_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_squad.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add mllm_rerank/cgcr_squad.py tests/test_cgcr_squad.py
git commit -m "feat(cgcr): squad orchestration (decompose/verify/critic) + recovery control"
```

---

### Task 5: Entry point + config + cross-import guard + README

**Files:**
- Create: `mllm_rerank/cgcr.py`
- Create: `mllm_rerank/cgcr_config.yaml`
- Create: `mllm_rerank/run_cgcr.sh`
- Modify: `mllm_rerank/README.md` (append a CGCR section)
- Test: `tests/test_cgcr_fusion_wiring.py`
- Reuse (no change): `tests/test_mllm_no_cross_import.py` already globs all `mllm_rerank/*.py` from the repo root, so it now also guards the new CGCR files — Step 9 re-runs it to confirm.

**Interfaces:**
- Consumes: `mllm_rerank.cmp_features` (`load_cmp_components`, `extract_cmp_features`, `compute_cmp_itm_scores`), `mllm_rerank.metrics` (`get_metrics`, `print_rs`), `mllm_rerank.fusion` (`fuse_scores`), `mllm_rerank.cgcr_gate` (`ambiguity_gate`), `mllm_rerank.claims` (`score_ground`), `mllm_rerank.cgcr_squad` (`decompose`, `verify_pairs`, `critic_pass`, `should_continue_recovery`), `mllm_rerank.squad` (`load_image`), `mllm_rerank.mllm` (`MLLMs`).
- **Adjudicator deferred:** `cgcr_squad.adjudicate` (built + unit-tested in Task 4) is NOT wired into `cgcr.py` in this first cut. A true listwise tie-break needs >1 image per prompt; the frozen `MLLMs` wrapper sends one image per prompt (`limit_mm_per_prompt={"image":1}`) and the Global Constraints forbid modifying `mllm.py`. The Adjudicator stays ready for a future multi-image wrapper method. `tie_margin` remains in config as a reserved knob.
- Produces:
  - `build_sem_dict(ground_scores: dict[tuple[int,int], float]) -> dict[tuple[int,int], float]` — identity pass-through helper kept tiny so `cgcr.py` can be unit-tested for the fusion-wiring contract: it maps CGCR `(query_i, local_j)` grounded scores into the `s_sem` dict shape that `fusion.fuse_scores` consumes. (This isolates one testable seam; the rest of `cgcr.py` is the entry `main()`, validated by `ast.parse`.)

- [ ] **Step 1: Write the failing test**

`tests/test_cgcr_fusion_wiring.py`:

```python
import torch
from mllm_rerank.cgcr import build_sem_dict
from mllm_rerank.fusion import fuse_scores


def test_build_sem_dict_passthrough():
    g = {(0, 1): 0.9, (0, 0): -0.2}
    out = build_sem_dict(g)
    assert out == g          # shape compatible with fuse_scores' s_sem


def test_cgcr_grounded_score_fuses_like_rerank():
    # grounded score promotes candidate at local_j=1 (gallery idx 2)
    s_str = torch.tensor([[0.8, 0.2, 0.5, 0.1]])
    topk_idx = torch.tensor([[0, 2]])
    g = {(0, 0): 0.0, (0, 1): 1.0}        # min-max over {0,1} -> {0,1}
    s_final = fuse_scores(s_str, topk_idx, build_sem_dict(g), lam=0.4)
    # local0->gallery0: 0.4*0.8+0.6*0 = 0.32 ; local1->gallery2: 0.4*0.5+0.6*1 = 0.80
    assert abs(s_final[0, 0].item() - 0.32) < 1e-5
    assert abs(s_final[0, 2].item() - 0.80) < 1e-5
    assert s_final[0, 1].item() == 0.2 and s_final[0, 3].item() == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_fusion_wiring.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mllm_rerank.cgcr'`.

- [ ] **Step 3: Implement `mllm_rerank/cgcr.py`**

```python
"""CGCR entry: Stage-1 -> ambiguity gate -> decompose -> recovery verify -> critic -> fuse -> adjudicate -> metrics."""
import os
import gc
import json
import time
import argparse
import torch
from ruamel.yaml import YAML

from mllm_rerank.cmp_features import (
    load_cmp_components, extract_cmp_features, compute_cmp_itm_scores,
)
from mllm_rerank.metrics import get_metrics, print_rs
from mllm_rerank.fusion import fuse_scores
from mllm_rerank.cgcr_gate import ambiguity_gate
from mllm_rerank.claims import score_ground
from mllm_rerank.cgcr_squad import (
    decompose, verify_pairs, critic_pass, should_continue_recovery,
)
from mllm_rerank.squad import load_image
from mllm_rerank.mllm import MLLMs

yaml = YAML(typ="safe")


def build_sem_dict(ground_scores):
    """Pass CGCR grounded (i, local_j) scores through to fuse_scores' s_sem shape."""
    return dict(ground_scores)


def _logger(out_dir):
    import logging
    os.makedirs(out_dir, exist_ok=True)
    lg = logging.getLogger("cgcr")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        sh = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(out_dir, "cgcr.log"))
        fmt = logging.Formatter("%(asctime)s %(message)s")
        sh.setFormatter(fmt)
        fh.setFormatter(fmt)
        lg.addHandler(sh)
        lg.addHandler(fh)
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f)
    out_dir = cfg["out_dir"]
    lg = _logger(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    table = cfg["category_weights"]
    default_w = cfg["default_weight"]

    # ---- Stage 1 (shared cache with rerank.py) ----
    cache = os.path.join(out_dir, "cmp_features.pt")
    if os.path.exists(cache):
        lg.info(f"Loading cached CMP features: {cache}")
        c = torch.load(cache)
        s_str = c["s_str"]
        img_paths = c["img_paths"]
        texts = c["texts"]
        qids = c["q_pids"]
        pids = c["g_pids"]
    else:
        from dataset import create_dataset, create_loader
        cmp_config, tokenizer, model = load_cmp_components(
            cfg["cmp_config"], cfg["cmp_checkpoint"], device)
        _, ds = create_dataset(cmp_config, evaluate=True)
        loader = create_loader([ds], [None], batch_size=[cmp_config["batch_size_test"]],
                               num_workers=[4], is_trains=[False], collate_fns=[None])[0]
        feats = extract_cmp_features(model, loader, tokenizer, device, cmp_config)
        s_str = compute_cmp_itm_scores(model, feats, device, cmp_config)
        img_paths = feats["img_paths"]
        texts = feats["texts"]
        qids = torch.tensor(ds.q_pids)
        pids = torch.tensor(ds.g_pids)
        torch.save({"s_str": s_str.cpu(), "img_paths": img_paths, "texts": texts,
                    "q_pids": qids, "g_pids": pids, "cmp_config": cmp_config}, cache)
        del model, tokenizer, loader, ds, feats
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    qids = torch.as_tensor(qids)
    pids = torch.as_tensor(pids)
    base_row = get_metrics(s_str.cpu(), qids, pids, "CMP-Base", False)
    lg.info(f"Base CMP: R1={base_row[1]:.2f} R5={base_row[2]:.2f} R10={base_row[3]:.2f} mAP={base_row[4]:.2f}")

    # ---- ambiguity gate ----
    gate_mask, margins = ambiguity_gate(s_str, cfg["gate_margin"])
    active = [i for i, g in enumerate(gate_mask) if g]
    lg.info(f"Ambiguity gate: {len(active)}/{len(gate_mask)} queries activated")

    # ---- load MLLM ----
    llm = MLLMs(cfg["model_dir"],
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.7),
                max_model_len=cfg.get("max_model_len", 1536))

    # ---- decompose (only active queries) ----
    active_texts = [texts[i] for i in active]
    claims_active = decompose(llm, active_texts, text_micro_batch=cfg.get("text_micro_batch", 16))
    claims_by_q = {i: claims_active[k] for k, i in enumerate(active)}

    # ---- recall-recovery verify + grounded score (store verdicts for the Critic) ----
    ground_scores = {}        # (i, local_j) -> grounded score; local_j indexes sorted_idx[i]
    verdicts_by_pair = {}     # (i, local_j) -> aggregated verdict list
    sorted_idx = torch.argsort(s_str, dim=1, descending=True)   # [Q, G]
    for i in active:
        claims = claims_by_q[i]
        if not claims:
            continue
        local_done = set()
        k = cfg["k0"]
        round_idx = 0
        while True:
            new_locals = [lj for lj in range(min(k, sorted_idx.size(1))) if lj not in local_done]
            work = [(claims, load_image(img_paths[int(sorted_idx[i, lj])])) for lj in new_locals]
            verds = verify_pairs(llm, work, n_samples=cfg["n_samples"],
                                 temperature=cfg.get("verifier_temperature", 0.6),
                                 image_micro_batch=cfg.get("image_micro_batch", 8))
            best = max([ground_scores[(i, lj)] for lj in local_done], default=-1.0)
            for lj, v in zip(new_locals, verds):
                local_done.add(lj)
                verdicts_by_pair[(i, lj)] = v
                sg = score_ground(claims, v, table, default_w)
                ground_scores[(i, lj)] = sg
                best = max(best, sg)
            nxt = should_continue_recovery(best, k, round_idx,
                                           hit_theta=cfg["hit_theta"], k_step=cfg["k_step"],
                                           k_max=cfg["k_max"], max_rounds=cfg["max_rounds"])
            if nxt is None:
                break
            k = nxt
            round_idx += 1

    # ---- critic pass on the suspect claims of each active query's best candidate (reuse stored verdicts) ----
    conf_floor = cfg.get("critic_conf_floor", 0.5)
    for i in active:
        claims = claims_by_q.get(i)
        if not claims:
            continue
        pairs = [(lj, ground_scores[(i, lj)]) for (qi, lj) in ground_scores if qi == i]
        if not pairs:
            continue
        best_lj = max(pairs, key=lambda t: t[1])[0]
        v_best = verdicts_by_pair[(i, best_lj)]
        suspect = []
        for c in claims:
            vv = next((x for x in v_best if x["id"] == c["id"]), None)
            if vv is None or vv["label"] == "CONTRADICTED" or vv["conf"] < conf_floor:
                suspect.append(c)
        if not suspect:
            continue
        crit = critic_pass(llm, suspect, load_image(img_paths[int(sorted_idx[i, best_lj])]),
                           image_micro_batch=cfg.get("image_micro_batch", 8))
        if crit:
            merged = {x["id"]: x for x in v_best}
            for x in crit:
                merged[x["id"]] = x
            ground_scores[(i, best_lj)] = score_ground(claims, list(merged.values()), table, default_w)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    # ---- fuse: topk_idx[i, local_j] = gallery index (sorted_idx prefix), aligned with ground_scores keys ----
    kk = min(cfg["k_max"], sorted_idx.size(1))
    topk_idx = sorted_idx[:, :kk].clone()
    s_final = fuse_scores(s_str, topk_idx, build_sem_dict(ground_scores), lam=cfg["lambda"])
    # (Adjudicator tie-break deferred — see header note; needs a multi-image wrapper.)

    # ---- metrics + save ----
    print_rs({"sims_base": s_str, "sims_cgcr": s_final}, qids, pids, lg)
    torch.save({"s_final": s_final.cpu(),
                "ground_scores": {f"{i}_{j}": v for (i, j), v in ground_scores.items()}},
               os.path.join(out_dir, "cgcr_result.pt"))
    lg.info("Done.")


if __name__ == "__main__":
    main()
```

Note: the Adjudicator tie-break is **deferred** (see the Task-5 header note): a true listwise comparison needs more than one image per prompt, which the frozen `MLLMs` wrapper does not support and the Global Constraints forbid changing. `cgcr.py`'s tested seam is `build_sem_dict` + `fuse_scores`; the grounded-score ranking is complete without the Adjudicator. Enabling it later requires a multi-image wrapper method, then a tie-break loop placed **before** `del llm`.

- [ ] **Step 4: Verify `cgcr.py` parses**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -c "import ast; ast.parse(open('mllm_rerank/cgcr.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Run the fusion-wiring test**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_fusion_wiring.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Create `mllm_rerank/cgcr_config.yaml`**

```yaml
# Stage-1 (shared with rerank.py)
cmp_config: 'configs/cmp.yaml'              # EDIT: trained CMP stage-1 yaml
cmp_checkpoint: 'output/cmp/best.pth'        # EDIT: trained CMP checkpoint
model_dir: 'checkpoint/Qwen3-VL-8B-Instruct' # EDIT: local MLLM weights
# vLLM
gpu_memory_utilization: 0.7
max_model_len: 1536
image_micro_batch: 8
text_micro_batch: 16
# Decomposer category -> weight (deterministic; MLLM only assigns category)
category_weights:
  action: 1.5
  anomaly: 1.5
  upper: 1.0
  lower: 1.0
  gender: 0.8
  accessory: 0.6
  hair: 0.5
  background: 0.3
default_weight: 0.7
# Verifier
n_samples: 3
verifier_temperature: 0.6
critic_conf_floor: 0.5
# Gate + recall recovery
gate_margin: 0.05
k0: 5
k_step: 5
k_max: 20
hit_theta: 0.6
max_rounds: 2
# Fusion + tie-break
lambda: 0.4
tie_margin: 0.03
out_dir: 'output/cgcr'
```

- [ ] **Step 7: Create `mllm_rerank/run_cgcr.sh`**

```bash
#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p output/cgcr logs
python -m mllm_rerank.cgcr --config mllm_rerank/cgcr_config.yaml \
    2>&1 | tee logs/cgcr_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 8: Append a CGCR section to `mllm_rerank/README.md`**

Append (do not rewrite the file):

````markdown

## CGCR — Claim-Grounded Comparative Rerank (alternative reranker)

A second, zero-shot reranker that improves on the SSDC Stage-2 idea. For each
query it decomposes the text into weighted atomic claims, verifies each claim
against each candidate image (graded signed entailment, self-consistency), runs a
targeted Critic on suspect claims, gates on Stage-1 ambiguity, and adaptively
deepens top-k when nothing matches. Score = weighted signed entailment, fused
with the structural score. (A listwise Adjudicator tie-break is designed but
deferred — it needs a multi-image MLLM wrapper.)

```bash
# edit mllm_rerank/cgcr_config.yaml paths first
bash mllm_rerank/run_cgcr.sh
```

CGCR does not replace `rerank.py`; both reuse the same `cmp_features.pt` Stage-1
cache so they can be compared on identical inputs. See
`docs/superpowers/specs/2026-06-28-cgcr-rerank-design.md` for strengths,
weaknesses, and the full SSDC comparison.
````

- [ ] **Step 9: Run the full CGCR + guard suite**

Run: `cd ~/Code/Project/Sim2Real-Track4 && .venv/bin/python -m pytest tests/test_cgcr_prompts.py tests/test_cgcr_claims.py tests/test_cgcr_gate.py tests/test_cgcr_squad.py tests/test_cgcr_fusion_wiring.py tests/test_mllm_no_cross_import.py -q`
Expected: PASS (all green; the no-cross-import guard now also covers the new `mllm_rerank/cgcr*.py` and `claims.py` files).

- [ ] **Step 10: Commit**

```bash
git add mllm_rerank/cgcr.py mllm_rerank/cgcr_config.yaml mllm_rerank/run_cgcr.sh \
        mllm_rerank/README.md tests/test_cgcr_fusion_wiring.py
git commit -m "feat(cgcr): entry pipeline + config/run + README + fusion-wiring test"
```

---

## Deferred Smoke Test (needs prereqs)

When Qwen3-VL-8B-Instruct, `vllm`, a trained CMP checkpoint, and PAB test data are
available on a GPU box:

```bash
# edit mllm_rerank/cgcr_config.yaml (cmp_config, cmp_checkpoint, model_dir)
bash mllm_rerank/run_cgcr.sh
```

Confirm: the log prints the **Base CMP** row, the ambiguity-gate activation count,
then a PrettyTable with `sims_base` vs `sims_cgcr`; `output/cgcr/cgcr_result.pt` is
saved. Report the accuracy delta vs base AND vs `rerank.py` (run both on the shared
cache), plus the average MLLM calls per activated query. (Future: add a multi-image
`MLLMs` method, then wire the deferred Adjudicator tie-break and re-measure.)
```
