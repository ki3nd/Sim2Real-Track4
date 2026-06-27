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
    # images for gated queries are pre-loaded here (eagerly) before the squad runs
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
