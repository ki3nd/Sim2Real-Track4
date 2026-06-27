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
