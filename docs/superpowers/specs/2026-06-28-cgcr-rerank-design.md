# CGCR — Claim-Grounded Comparative Rerank (MLLM) — Design Spec

**Date:** 2026-06-28
**Scope:** A second MLLM rerank entry point in the existing `mllm_rerank/`
package that improves on the SSDC Stage-2 Detective Squad. CGCR reranks the
CMP Stage-1 candidates by **decomposing the query into weighted atomic claims**,
**visually verifying each claim against each candidate image** (graded
entailment, not text-text cosine), with a **self-consistency Verifier + targeted
Critic**, **ambiguity-based gating**, **adaptive recall recovery** (deepen top-k
when nothing matches), and a **listwise Adjudicator** tie-break. Zero-shot
Qwen3-VL-8B-Instruct via vLLM. Eval-only on PAB test; produces a metrics table.
**Out of scope:** any fine-tuning / LoRA / SFT, hard-negative mining, changes to
Stage 1, the existing semantic-cosine rerank (`mllm_rerank/rerank.py`) which stays
as-is for comparison.

## 1. Goal

Lift rerank quality and robustness over SSDC Stage-2 by fixing its structural
weaknesses (recall ceiling, cascade hallucination, text-text cosine compression,
absolute-threshold gating) — while staying **challenge-compliant** (public model,
zero-shot, no hard sampling, no test-distribution training). CGCR is a sibling of
`rerank.py`, reusing the same package infrastructure (`MLLMs`, `metrics`,
`cmp_features`, `fusion` helpers) so the two can be A/B compared on identical
Stage-1 inputs.

## 2. The four design shifts (each targets a named SSDC weakness)

| Shift | SSDC weakness it fixes |
|---|---|
| Score by **visual entailment over atomic claims** (image↔query), graded, with **negative penalty for contradictions** | text-text cosine compresses discriminability; loses fine signal |
| **Listwise Adjudicator** tie-break (compare candidates side-by-side) on near-ties | pointwise scoring can't separate look-alike candidates |
| **Adaptive recall recovery** — deepen top-k when no candidate matches | hard recall ceiling: target outside top-k is unrecoverable |
| **Ambiguity gating** — activate on score margin/entropy, not absolute `S_str>ξ` | absolute-threshold gate serves the easy cases of the hard problem |
| **Critic + self-consistency** instead of a one-way pipeline | cascade error propagation with no self-correction |

## 3. Agents (one zero-shot MLLM, role-switched by prompt)

All agents are the SAME Qwen3-VL-8B-Instruct served once via the existing
`mllm_rerank.mllm.MLLMs`. Roles differ only by prompt + which modality is sent.

- **Decomposer** (text-only, **once per query**, cacheable): split query `T` into
  atomic claims with categories and weights. Output strict JSON:
  ```json
  {"claims":[
    {"id":1,"category":"gender","text":"male","weight":0.8},
    {"id":2,"category":"upper","text":"red shirt","weight":1.0},
    {"id":3,"category":"action","text":"falling down","weight":1.5},
    {"id":4,"category":"background","text":"playground","weight":0.3}
  ]}
  ```
  **Action / anomaly-behavior claims get the highest weight** (the task's crux);
  background/lighting the lowest. Weights come from a fixed category→weight table
  in config (the MLLM only assigns categories + extracts claim text; the spec does
  NOT trust the MLLM to invent weights — it maps category→weight deterministically
  after parsing). Unknown category → default weight.

- **Verifier** (vision, run per candidate image, **self-consistency N samples**):
  given the image + the claim list, for each claim output a verdict + confidence +
  one-line evidence:
  ```json
  {"verdicts":[
    {"id":1,"label":"ENTAILED","conf":0.9,"evidence":"adult male build"},
    {"id":3,"label":"CONTRADICTED","conf":0.8,"evidence":"person is doing push-ups, not falling"}
  ]}
  ```
  Labels: `ENTAILED` / `NEUTRAL` (not observable) / `CONTRADICTED`. Run N
  independent samples (temperature > 0); aggregate per claim by **majority label**,
  confidence = mean conf of the majority label. (N=3 default.)

- **Critic** (vision, conditional): re-examine only claims that came back
  `CONTRADICTED` or with `conf < critic_conf_floor` on the high-scoring candidates.
  Single focused prompt per (candidate, suspect-claim-set): "Look again ONLY at
  whether <claims>. Confirm or overturn each." Overwrites those verdicts. Guards
  against both false-contradiction (kills a true match) and false-entail
  (hallucinated match).

- **Adjudicator** (vision, listwise, conditional): when the top-2/3 candidates are
  within `tie_margin` of each other after fusion, send the query + those 2-3 images
  together and ask for a relative ranking focused on the highest-weight
  (action/anomaly) claims. Its order overrides the fused order for those positions
  only.

## 4. Scoring

Per candidate `j`, per claim `i` (after Verifier aggregation, then Critic
overrides):
```
v_ij = +conf_ij   if ENTAILED
        0          if NEUTRAL          # unobservable: no reward, no penalty
       -conf_ij    if CONTRADICTED     # hard penalty
S_ground(j) = ( Σ_i w_i · v_ij ) / ( Σ_i w_i )        ∈ [-1, 1]
```
- NEUTRAL is **not** penalized → robust to occlusion / low-res surveillance frames
  (cannot see the shoes ≠ wrong shoes).
- A candidate that matches appearance but **contradicts the action claim** is
  pushed down hard (action weight is largest) — the exact Pose-Semantic-Gap case.

Fusion with the Stage-1 structural score (reuse `fusion.fuse_scores` semantics):
```
S_ground_norm = min-max over processed (i,j)            # same normalization style as rerank.py
S_fused(j)    = lambda * S_str_norm(j) + (1 - lambda) * S_ground_norm(j)
```
Then the Adjudicator reorders near-ties (positions where
`|S_fused(top) - S_fused(next)| < tie_margin`).

## 5. Algorithm flow

```
1. Stage-1 (reuse mllm_rerank.cmp_features): S_str + img_paths + texts + qids/pids
   (cache cmp_features.pt; identical to rerank.py so both share the cache).
2. Ambiguity gate (per query i):
     margin_i = S_str_sorted[i,0] - S_str_sorted[i,1]   (or normalized entropy of top-m)
     if margin_i >= gate_margin:  SKIP MLLM  (trust Stage-1; row keeps S_str_norm)
     else:                        activate CGCR for query i
3. Decompose: claims_i = Decomposer(texts[i])   (text-only; cache per query)
4. Recall-recovery loop (k = k0; up to R rounds):
     verify newly-exposed top-k candidates with Verifier (self-consistency N)
       -> S_ground for those (i,j)
     if max_j S_ground(i,j) >= hit_theta:   break          # confident match found
     elif k < k_max:   k += k_step;  continue               # DEEPEN top-k (recall recovery)
     else:             break
5. Critic pass on suspect claims of the top-scoring candidates per query.
6. Fuse -> Adjudicator tie-break on near-ties -> final ranking.
7. Metrics: print_rs({sims_base, sims_cgcr}); save S_final + per-claim verdicts JSON.
```

## 6. Files (extend `mllm_rerank/`, reuse existing modules)

- **Create `mllm_rerank/cgcr_prompts.py`** — `DECOMPOSER_PROMPT`,
  `VERIFIER_PROMPT`, `CRITIC_PROMPT`, `ADJUDICATOR_PROMPT` + parsers
  (`parse_claims`, `parse_verdicts`, `parse_adjudication`) with JSON + fallback,
  mirroring `prompts.py` style.
- **Create `mllm_rerank/claims.py`** — `category_weight(category, table, default)`;
  `aggregate_self_consistency(samples) -> per-claim majority verdict`;
  `score_ground(claims, verdicts) -> float` (the Section-4 formula).
- **Create `mllm_rerank/cgcr_squad.py`** — orchestration:
  `decompose(llm, queries)`, `verify(llm, claims, cand_imgs, n_samples)`,
  `critic(llm, ...)`, `adjudicate(llm, query, images)`; plus the recall-recovery
  loop helper.
- **Create `mllm_rerank/cgcr_gate.py`** — `ambiguity_gate(s_str, gate_margin,
  top_m) -> (gate_mask, margins)`.
- **Create `mllm_rerank/cgcr.py`** — entry `__main__` wiring the flow (the CGCR
  analogue of `rerank.py`); reuse `cmp_features`, `metrics`, and `fusion.fuse_scores`.
- **Create `mllm_rerank/cgcr_config.yaml`** — Section-7 knobs.
- **Reuse unchanged:** `mllm.py` (`MLLMs`, `batch_infer`), `metrics.py`,
  `cmp_features.py`, `fusion.py` (`fuse_scores`; `build_topk_and_gate` only for the
  initial k0 slice).
- **Constraint:** no file under `mllm_rerank/` references `open-sources` or mutates
  `sys.path` (same guard test extends to the new files).

## 7. Config (`mllm_rerank/cgcr_config.yaml`)

```yaml
# Stage-1 (shared with rerank.py)
cmp_config: 'configs/cmp.yaml'
cmp_checkpoint: 'output/cmp/best.pth'
model_dir: 'checkpoint/Qwen3-VL-8B-Instruct'
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
n_samples: 3            # self-consistency
verifier_temperature: 0.6
critic_conf_floor: 0.5  # claims below this get a Critic pass
# Gate + recall recovery
gate_margin: 0.05       # skip MLLM if top1-top2 S_str margin >= this
gate_top_m: 5
k0: 5
k_step: 5
k_max: 20
hit_theta: 0.6          # S_ground threshold that ends recall recovery
max_rounds: 2
# Fusion + tie-break
lambda: 0.4
tie_margin: 0.03
out_dir: 'output/cgcr'
```

## 8. Strengths

- **Grounded, discriminative score.** Verdicts are image↔query entailment, graded
  and signed. Contradiction penalties separate "right clothes, wrong action" from a
  true match — the precise failure SSDC's cosine cannot express. Wider score
  dynamic range than text-text cosine.
- **Action-weighted.** The anomaly/action claim dominates the score, matching the
  task definition (behavioral retrieval), not appearance bias.
- **Recall recovery raises the ceiling.** Deepening top-k when nothing matches
  recovers targets that sit just outside the SSDC top-k0 window — without
  retraining Stage 1.
- **Ambiguity gating spends compute where it helps.** Activates on flat/ambiguous
  Stage-1 distributions (where reranking changes the answer), skips peaked ones.
- **Self-correction.** Self-consistency + a targeted Critic reduce single-pass
  hallucination, the dominant SSDC failure mode.
- **Interpretable + auditable.** Per-claim verdicts with one-line evidence explain
  every rank change — valuable for an anomaly/surveillance setting.
- **Challenge-compliant.** Public model, zero-shot, no hard mining, no
  test-distribution training. (SSDC's best config needs LoRA SFT on a mined
  hard-negative set.)

## 9. Weaknesses / risks

- **Higher inference cost.** Self-consistency ×N, recall recovery ×R rounds, plus
  Adjudicator on ties. Mitigated by N=3, R≤2, Adjudicator only on near-ties, and
  the gate culling easy queries — but per-activated-query cost exceeds SSDC
  single-pass. Must measure throughput at PAB scale.
- **Decomposer is a new single point of failure.** Wrong claim split (e.g.,
  dropping the action) corrupts the whole score. Lower-risk than SSDC's Writer
  (text-only, cacheable, inspectable) but still a dependency.
- **Recall recovery is bounded.** Targets beyond `k_max` remain unrecoverable —
  better than SSDC (k0 only) but not a full fix for Stage-1 recall.
- **Verifier calibration.** `conf` from a zero-shot MLLM may be poorly calibrated;
  signed scoring amplifies overconfident contradictions. Self-consistency + Critic
  dampen this, not eliminate it.
- **More hyperparameters** (`gate_margin`, `hit_theta`, `tie_margin`, weights, N,
  R). In Sim2Real (synthetic train, real test) there is no in-distribution
  validation set, so these are tuned blind — overfitting risk on choices. Spec ships
  conservative defaults and recommends sensitivity logging.
- **Latency vs the "real-time surveillance" motivation** still unresolved (inherited
  from any MLLM-in-the-loop design).

## 10. Comparison with SSDC Stage-2

| Dimension | SSDC Stage-2 (Detective Squad) | CGCR (this spec) |
|---|---|---|
| Agents | Detective(Yes/No) → Analyst(15-item checklist) → Writer(new caption) | Decomposer → Verifier(self-consistency) → Critic → Adjudicator |
| Score signal | paper: `cos(E_txt(T), E_txt(T_new))`; released code: hard-boost verified index to 1.0 | weighted signed visual entailment per atomic claim, image↔query |
| Candidate separation | pointwise, independent per candidate | pointwise grounded score **+ listwise Adjudicator** on ties |
| Negative evidence | none (caption only describes; cosine ≥ 0 region) | explicit **contradiction penalty** (signed) |
| Occlusion handling | implicit (Writer may hallucinate to fill gaps) | explicit **NEUTRAL** (no reward/penalty) |
| Recall beyond top-k | fixed top-k0; target outside ⇒ lost | **adaptive deepening** to `k_max` |
| Gate | `S_str > ξ` (absolute structural score) | **margin/entropy** (Stage-1 ambiguity) |
| Hallucination control | one-way checklist | self-consistency N + targeted Critic |
| Training needed for SOTA | LoRA SFT on mined `D_sft` (hard negatives) | **none (zero-shot)** |
| Challenge compliance | hard-mining step is questionable under rules | compliant |
| Compute per activated query | single pass (cheapest) | higher (N×, R×, Adjudicator) |
| Interpretability | a generated caption | per-claim verdict + evidence + signed contribution |
| Reuse in this repo | `mllm_rerank/rerank.py` (already merged) | new `mllm_rerank/cgcr.py`, shares `MLLMs`/`metrics`/`cmp_features`/`fusion` |

**Relationship:** CGCR does not replace `rerank.py`; it is a second entry point so
both can be evaluated on the same cached `S_str`. The honest expectation: CGCR
trades more compute for better discrimination and recall recovery; whether the
accuracy gain justifies the cost is an empirical question the deferred smoke must
answer.

## 11. Testing (CPU, no model — same discipline as the merged module)

- **Decomposer/Verifier/Adjudicator parsers:** valid JSON → structured objects;
  malformed JSON → safe fallback (empty verdicts / raw text); extra prose around
  JSON tolerated.
- **`category_weight`:** known category → table value; unknown → default.
- **`aggregate_self_consistency`:** N sample verdicts per claim → majority label +
  mean-of-majority conf; ties broken deterministically (toward NEUTRAL).
- **`score_ground`:** ENTAILED→positive, CONTRADICTED→negative, NEUTRAL→0;
  action-weight dominance verified; all-NEUTRAL → 0.
- **`ambiguity_gate`:** peaked row (large margin) → gated out; flat row → activated.
- **recall-recovery loop (mock llm):** stops on `hit_theta`; deepens k when below;
  caps at `k_max` / `max_rounds`.
- **Adjudicator tie-break (mock):** reorders only positions within `tie_margin`;
  leaves clear gaps untouched.
- **no-cross-import guard:** extend the existing grep test to the new files.
- **Smoke (DEFERRED — needs Qwen3-VL-8B + vLLM + GPU + CMP ckpt + PAB test):**
  run `cgcr.py`; base-vs-cgcr table prints; per-claim verdict JSON saved; report
  the accuracy delta and the average MLLM calls/query vs `rerank.py`.

## 12. Prerequisites (user-managed)

Same as `rerank.py`: Qwen3-VL-8B-Instruct weights, `vllm` + `qwen_vl_utils`, a
trained CMP Stage-1 checkpoint + yaml, PAB test data + pose (if the Stage-1 yaml
uses `be_pose_img`), `bert-base-uncased` for the CMP text tower (used only for the
shared Stage-1 cache, not for CGCR scoring).
