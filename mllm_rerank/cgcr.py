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
