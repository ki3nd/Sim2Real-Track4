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
    cmp_cfg["load_params"] = False
    cmp_cfg["be_hard"] = False

    with open(args.lhp_config) as f:
        lhp_cfg = YAML(typ="safe").load(f)

    _, test_ds = create_dataset(cmp_cfg, evaluate=True)
    cmp_loader = create_loader(
        [test_ds],
        [None],
        batch_size=[cmp_cfg["batch_size_test"]],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None],
    )[0]
    n_query, n_gallery = len(test_ds.text), len(test_ds.image)

    bert_tok = BertTokenizer.from_pretrained(cmp_cfg["text_encoder"])
    cmp_model = Search(config=cmp_cfg)
    cmp_model.load_pretrained(args.cmp_checkpoint)
    cmp_model = cmp_model.to(device).eval()
    _, image_embeds, text_embeds, text_atts = evaluation_itc(
        cmp_model, cmp_loader, bert_tok, device, cmp_cfg
    )

    lhp_model = load_retriever(
        args.lhp_kind, args.lhp_checkpoint, lhp_cfg["drop_path_rate"], device
    )
    spm_tok = XLMRobertaTokenizer(lhp_cfg["spm_model"])
    lhp_tf = LHPTransform(
        lhp_cfg["resolution"], tuple(lhp_cfg["crop_scale"]), local_prob=0.0
    )
    img_feats = encode_images(
        lhp_model,
        test_ds.ann,
        test_ds.image_root,
        lhp_tf,
        lhp_cfg["batch_size_eval"],
        device,
    )
    txt_feats = encode_texts(
        lhp_model,
        test_ds.text,
        spm_tok,
        lhp_cfg["max_tokens"],
        lhp_cfg["batch_size_eval"],
        device,
    )
    lhp_sims = similarity(img_feats, txt_feats)
    assert_aligned(lhp_sims, n_query, n_gallery)

    score = evaluation_itm(
        cmp_model,
        device,
        cmp_cfg,
        eval_args(),
        lhp_sims,
        image_embeds,
        text_embeds,
        text_atts,
    )
    print(
        f"=== LHP({args.lhp_kind}) top-{cmp_cfg['k_test']} -> CMP rerank | "
        f"{n_gallery} gallery, {n_query} queries ==="
    )
    mAP(score, test_ds.g_pids, test_ds.q_pids)


if __name__ == "__main__":
    main()
