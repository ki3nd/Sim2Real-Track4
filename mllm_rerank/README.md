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
