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
