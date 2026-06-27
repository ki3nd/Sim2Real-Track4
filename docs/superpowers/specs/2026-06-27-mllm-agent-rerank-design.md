# MLLM Agent Rerank (SSDC Stage-2) — Design Spec

**Date:** 2026-06-27
**Scope:** A standalone `mllm_rerank/` module that adds an **MLLM Detective-Squad
rerank stage** (SSDC paper Stage 2) on top of the repo's CMP Stage-1 retriever.
For each text query, it runs a **single-pass** Detective→Analyst→Writer agent
chain (Qwen3-VL-8B-Instruct, zero-shot, served via vLLM) over the top-k image
candidates, generates a new caption `T_new` per surviving candidate, and reranks
by fusing the CMP structural score with a **semantic cosine** `cos(E_txt(T),
E_txt(T_new))` (paper Eq. 2-4). Eval-only on PAB test; produces a metrics table.
**Out of scope:** LoRA SFT of the MLLM (zero-shot only), multi-round iteration
(single-pass only), hard-negative mining, training any model, LHP integration.

## 1. Goal

Reproduce SSDC Stage 2 as an eval-time rerank stage, following the **paper's
semantic-cosine scoring** (not the released SSDC code's hard-boost-index
shortcut). Reuse the repo's existing CMP `Search` model for Stage-1 structural
scores and as the frozen text encoder for `S_sem`. Port (vendor) the needed
helpers from `open-sources/SSDC` into this module — **never import across the
`open-sources/SSDC` directory** (no `sys.path` injection, no cross-repo import).

## 2. Hard constraint — no cross-repo import

All reused SSDC logic (`MLLMs` vLLM wrapper, the 3 prompts, `process_cap_`,
`extract_cmp_features`, `compute_cmp_itm_scores`, ranking/metric helpers) is
**copied and adapted into `mllm_rerank/`**. The module must not:
- add `open-sources/SSDC` (or `../../CMP`) to `sys.path`,
- `import` any symbol from a path under `open-sources/`.
It imports only from this repo (`models.model_search.Search`, `dataset.*`,
`models.*`) and third-party packages (`vllm`, `transformers`, `torch`,
`qwen_vl_utils`, `ruamel.yaml`, `prettytable`).

## 3. Challenge compliance

- Qwen3-VL-8B-**Instruct**, **zero-shot** (no fine-tuning) → public model, no
  training, no hard-negative mining, no test-distribution data used for
  training. The MLLM only reads test images at inference to rerank — allowed.
- CMP Stage-1 is the already-trained retriever in this repo (no change).

## 4. Architecture & data flow (per query, single-pass)

```
Stage-1 (reuse repo CMP Search, port SSDC extract/score helpers):
  extract_cmp_features(Search, loader, tokenizer)  -> text_feats, image_feats,
                                                       text_embeds, image_embeds, text_atts, img_paths, texts
  compute_cmp_itm_scores(...)                       -> S_str  [Q x G], min-max normalized
  cache to {out_dir}/cmp_features.pt (compute once)

Per query i:
  topk_idx = S_str[i].topk(k)                        # k candidates
  gate: skip squad if S_str[i].max() <= xi           # Eq.4 threshold gate
  for each candidate j in topk_idx:
    Detective: prompt1(image_j, T_i)  -> "Yes"/"No"  # "No" => candidate dropped
    Analyst:   prompt2(image_j)       -> 15-item checklist -> process_cap_()
    Writer:    prompt3(T_i + checklist) -> T_new_j    # JSON {"caption": ...}
    S_sem[i,j] = cosine( E_txt(T_i), E_txt(T_new_j) ) # CMP BERT tower, frozen
  # min-max normalize S_sem over processed (i,j) positions
  fuse only on processed top-k positions (Eq.4):
    S_final[i,j] = lambda * S_str_norm[i,j] + (1-lambda) * S_sem_norm[i,j]
  unprocessed positions keep S_str_norm
rank by S_final -> R@1/R@5/R@10/mAP (print base CMP vs +MLLM rerank)
```

- **`S_sem` is text-text cosine** between the original query and the Writer's new
  caption for that candidate — gallery-independent, varies per candidate, this is
  the ranking signal. Embed both via the **frozen CMP BERT text tower**
  (`model.get_text_embeds` → `model.get_text_feat`, L2-normalize), same space as
  `S_str`.
- Candidates dropped by the Detective (or never processed because
  `S_str.max() <= xi`) keep their `S_str_norm` value (no `S_sem`).

## 5. Files (all under `mllm_rerank/`)

- **`mllm_rerank/__init__.py`** — package marker.
- **`mllm_rerank/mllm.py`** — `class MLLMs`: vLLM wrapper for Qwen3-VL-8B
  (ported from SSDC `vllm_infer_SSDC.py`): `__init__(model_dir)` loads `LLM(...)`
  + `AutoProcessor`; `generate_response_multi_images(questions, images, sys, t)`
  (1 image per prompt) and `generate_response_text(questions, sys, t)`
  (text-only, for Writer); plus batched `batch_infer` / `batch_infer_txt`
  helpers (micro-batching, OOM-safe try/except → `[""]*n`).
- **`mllm_rerank/prompts.py`** — the three prompt strings `DETECTIVE_PROMPT`
  (Yes/No), `ANALYST_PROMPT` (15-item checklist), `WRITER_PROMPT` (aggregate →
  JSON `{"caption": ...}`), copied verbatim from SSDC `round_llm`; plus
  `process_checklist(raw_answers) -> List[List[str]]` (ported `process_cap_`:
  split on newline, drop `': '` lines, strip leading `N.` index, strip
  `Yes,`/`No,`, capitalize, ensure trailing `.`) and
  `parse_writer_caption(raw) -> str` (extract `"caption"` from the JSON; fall
  back to the raw text if JSON parse fails).
- **`mllm_rerank/squad.py`** — single-pass orchestration:
  `run_squad(llm, queries, topk_img_paths, gate_mask) -> {(i,j): T_new}`.
  Implements Detective gate → Analyst → Writer over the top-k candidates,
  batched through `MLLMs`. Returns the new caption per surviving `(query,
  candidate)` pair (and which candidates the Detective dropped).
- **`mllm_rerank/cmp_features.py`** — ported CMP Stage-1 helpers:
  `load_cmp_components(config_path, ckpt_path, device)` (build repo
  `models.model_search.Search`, `model.load_pretrained`, `BertTokenizer`),
  `extract_cmp_features(...)`, `compute_cmp_itm_scores(...)`,
  `embed_texts(model, tokenizer, texts, device, config) -> [N, D]` L2-normalized
  (used for both `T` and `T_new`). Adapted from SSDC but importing this repo's
  `Search`/`dataset` (no cross-repo path).
- **`mllm_rerank/metrics.py`** — ported `rank` / `get_metrics` / `print_rs`
  (R@1/5/10, mAP, mINP, rSum via PrettyTable).
- **`mllm_rerank/rerank.py`** — entry script (`__main__`): parse args + load
  YAML, run Stage-1 (or load cache), build top-k + gate, `run_squad`, compute
  `S_sem`, fuse (Eq.4), rank, print base-vs-rerank table, save `S_final` + the
  generated captions to `out_dir`.
- **`mllm_rerank/config.yaml`** — knobs (Section 7).
- **`mllm_rerank/run.sh`** — example invocation (single GPU, vLLM env vars).
- **`mllm_rerank/README.md`** — usage + the paper-vs-SSDC-code difference note.

## 6. Fusion detail (paper Eq. 4, adapted to single-pass)

- `S_str` from `compute_cmp_itm_scores` is already min-max normalized to `[0,1]`
  over the full matrix (call it `S_str_norm`).
- `S_sem` raw is cosine in `[-1,1]`. Min-max normalize it **over the set of
  processed `(i,j)` positions** to `[0,1]` (`S_sem_norm`); if fewer than 2
  processed positions, skip normalization (use raw, clamped to `[0,1]`).
- Threshold gate: a query participates only if `S_str_norm[i].max() > xi`;
  otherwise its row is unchanged.
- For each processed `(i,j)`: `S_final[i,j] = lambda*S_str_norm[i,j] +
  (1-lambda)*S_sem_norm[i,j]`. All other positions = `S_str_norm`.
- Default `lambda = 0.4`, `xi = 0.1` (matching SSDC `run_stage2.sh`); `k`
  (top-k candidates per query) default `10`. All in config.

## 7. Config (`mllm_rerank/config.yaml`)

```yaml
# Stage-1 CMP (repo Search model)
cmp_config: 'configs/<stage1>.yaml'        # repo CMP/SSDC stage-1 yaml
cmp_checkpoint: 'output/<stage1>/best.pth'
# MLLM
model_dir: 'checkpoint/Qwen3-VL-8B-Instruct'
gpu_memory_utilization: 0.7
max_model_len: 1536
image_micro_batch: 8
text_micro_batch: 16
temperature: 0.01
# Rerank
top_k: 10
xi: 0.1                # threshold gate on normalized S_str
lambda: 0.4            # fusion weight (structural vs semantic)
max_tokens_text: 56    # for CMP BERT embedding of T / T_new
out_dir: 'output/mllm_rerank'
```

## 8. Error handling

- vLLM generation wrapped in try/except → returns `[""]*n` on failure (ported
  from SSDC); empty Detective output treated as "No"; empty Writer caption →
  candidate keeps `S_str_norm` (no `S_sem` contribution).
- `parse_writer_caption` falls back to raw text if JSON is malformed.
- Stage-1 feature cache (`cmp_features.pt`): load if present, else compute and
  save (CMP model released from GPU before loading vLLM — port the
  `del model; empty_cache(); sleep` pattern so both fit on one GPU).

## 9. Testing

- **prompts/parse (CPU, no model):** `process_checklist` on a sample 15-line
  Qwen answer → list of cleaned sentences (index stripped, capitalized, trailing
  `.`); `parse_writer_caption` extracts `"caption"` from valid JSON and falls
  back on malformed JSON.
- **fusion (CPU, synthetic tensors):** given a small `S_str_norm`, a dict of
  `S_sem` for a few `(i,j)`, `xi`, `lambda` → `S_final` equals
  `lambda*str+(1-lambda)*sem` on processed positions and `str` elsewhere;
  rows with `max(S_str_norm) <= xi` are unchanged; ranking of a planted
  high-`S_sem` candidate improves.
- **squad single-pass (mock LLM):** a fake `MLLMs` returning canned
  "Yes"/checklist/JSON → `run_squad` drops "No" candidates and returns one
  `T_new` per surviving `(i,j)`.
- **metrics:** `get_metrics` on a known 10×10 similarity returns expected
  R@1/R@5/R@10 (port the existing test shape used elsewhere in the repo).
- **no-cross-import guard:** a test greps `mllm_rerank/*.py` asserting no line
  contains `open-sources` or adds it to `sys.path`.
- **Smoke (DEFERRED — needs Qwen3-VL-8B + vLLM + GPU + CMP ckpt + PAB test):**
  run `rerank.py` on a tiny gallery; the base-vs-rerank table prints and
  `S_final` is saved. Document when prereqs exist.

## 10. Prerequisites (user-managed)

Qwen3-VL-8B-Instruct weights at `model_dir`; `vllm` + `qwen_vl_utils` installed;
a trained CMP Stage-1 checkpoint + its yaml; PAB test data + pose (if the
Stage-1 yaml uses `be_pose_img`); `bert-base-uncased` for the CMP text tower.
