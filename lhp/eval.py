import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML
from PIL import Image
from transformers import XLMRobertaTokenizer

from lhp.transform import LHPTransform
from lhp.tokenization import tokenize_caption
from lhp.model import LHPRetriever
from lhp.infer import similarity
from dataset.utils import read_json_to_list, pre_caption
from eval import mAP  # reuse CMP scoring (single source of truth)


def build_test_index(ann, max_words):
    """CMP-format test records -> (g_pids, captions, q_pids).
    Gallery = images (g_pids = each image's image_id);
    queries = all captions flattened (q_pids = owning image's id)."""
    g_pids = [a["image_id"] for a in ann]
    captions, q_pids = [], []
    for a in ann:
        for c in a["caption"]:
            q_pids.append(a["image_id"])
            captions.append(pre_caption(c, max_words))
    return g_pids, captions, q_pids


class _ImageDataset(Dataset):
    """Yields gallery images in annotation order (so features align with g_pids)."""
    def __init__(self, ann, data_root, transform):
        self.ann, self.data_root, self.transform = ann, data_root, transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.data_root, self.ann[i]["image"])).convert("RGB")
        return self.transform(img)[0]  # global-view tensor (local_prob=0.0)


@torch.no_grad()
def encode_images(model, ann, data_root, transform, bs, device):
    loader = DataLoader(_ImageDataset(ann, data_root, transform),
                        batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    feats = [model.encode_image(batch.to(device)) for batch in loader]
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(model, captions, tokenizer, max_tokens, bs, device):
    feats = []
    for i in range(0, len(captions), bs):
        toks = [tokenize_caption(tokenizer, c, max_tokens) for c in captions[i:i + bs]]
        ids = torch.tensor([t[0] for t in toks], device=device)
        mask = torch.tensor([t[1] for t in toks], device=device)
        feats.append(model.encode_text(ids, mask))
    return torch.cat(feats, dim=0)


def load_retriever(kind, ckpt_path, drop_path_rate, device):
    if kind == "beit3":
        model = LHPRetriever(ckpt_path=ckpt_path, drop_path_rate=drop_path_rate)
    elif kind == "lhp":
        model = LHPRetriever(ckpt_path=None, drop_path_rate=drop_path_rate)
        state = torch.load(ckpt_path, map_location="cpu")["model"]
        msg = model.load_state_dict(state, strict=False)
        print("lhp ckpt -> missing:", msg.missing_keys, "unexpected:", msg.unexpected_keys)
    else:
        raise ValueError(f"--kind must be 'beit3' or 'lhp', got {kind!r}")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lhp/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", choices=["beit3", "lhp"], required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = YAML(typ="safe").load(f)
    device = torch.device("cuda")
    data_root = cfg["data_root"]

    ann = read_json_to_list(os.path.join(data_root, cfg["test_file"]))
    g_pids, captions, q_pids = build_test_index(ann, cfg["max_words"])

    model = load_retriever(args.kind, args.checkpoint, cfg["drop_path_rate"], device)
    tokenizer = XLMRobertaTokenizer(cfg["spm_model"])
    transform = LHPTransform(cfg["resolution"], tuple(cfg["crop_scale"]), local_prob=0.0)

    img_feats = encode_images(model, ann, data_root, transform, cfg["batch_size_eval"], device)
    txt_feats = encode_texts(model, captions, tokenizer, cfg["max_tokens"], cfg["batch_size_eval"], device)
    # infer.similarity(img_feats, txt_feats) == txt_feats @ img_feats.T == [N_query, N_gallery]
    scores_t2i = similarity(img_feats, txt_feats).cpu().numpy()  # [N_query, N_gallery]

    print(f"=== eval kind={args.kind} ckpt={args.checkpoint} | "
          f"{len(ann)} gallery, {len(captions)} queries ===")
    mAP(scores_t2i, g_pids, q_pids)


if __name__ == "__main__":
    main()
