import os
import math
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ruamel.yaml import YAML
from transformers import BertTokenizer

from models.uit import UIT
from uit.xvlm_init import load_xvlm
from dataset.search_dataset import search_train_dataset, TextMaskingGenerator


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def build_masked_text(captions, text_input, tokenizer, device, mask_gen, max_masks, max_tokens):
    """Mirror of root train.py mlm(): build (text_ids_masked, masked_pos, masked_ids)."""
    masked = tokenizer(captions, padding="max_length", truncation=True,
                       max_length=max_tokens, return_tensors="pt").to(device)
    ids_masked = masked.input_ids
    masked_pos = torch.zeros((ids_masked.shape[0], max_masks), dtype=torch.int64, device=device)
    masked_ids = torch.full((ids_masked.shape[0], max_masks), -100, dtype=torch.long, device=device)
    for idx, tid in enumerate(ids_masked):
        tid_m, pos = mask_gen(tid)
        mids = [text_input.input_ids[idx][p].item() for p in pos]
        n = len(pos)
        masked_pos[idx, :n] = torch.tensor(pos, dtype=torch.int64, device=device)
        masked_ids[idx, :n] = torch.tensor(mids, dtype=torch.long, device=device)
    return ids_masked, masked_pos, masked_ids


def train_transform(res):
    norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
    return transforms.Compose([
        transforms.Resize((res, res), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="uit/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = YAML(typ="safe").load(f)

    local_rank, rank, world = setup_ddp()                 # BEFORE building model (ITC gather rank/world)
    device = torch.device("cuda", local_rank)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(cfg["text_encoder"])
    mask_gen = TextMaskingGenerator(tokenizer, cfg["mask_prob"], cfg["max_masks"],
                                    cfg["skipgram_prb"], cfg["skipgram_size"], cfg["mask_whole_word"])
    dataset = search_train_dataset(cfg, train_transform(cfg["swin"]["img_size"]))

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    per_gpu_bs = cfg["batch_size"] // world
    loader = DataLoader(dataset, batch_size=per_gpu_bs, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UIT(cfg)
    load_xvlm(model, cfg["xvlm_ckpt"])
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = math.ceil(len(dataset) / cfg["batch_size"]) * cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        for i, (image, caption, *_rest) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            text = tokenizer(list(caption), padding="max_length", truncation=True,
                             max_length=cfg["max_tokens"], return_tensors="pt").to(device)
            ids_masked, masked_pos, masked_ids = build_masked_text(
                list(caption), text, tokenizer, device, mask_gen, cfg["max_masks"], cfg["max_tokens"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(image, text.input_ids, text.attention_mask,
                            ids_masked, masked_pos, masked_ids)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            scheduler.step()
            if rank == 0 and i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(loader)} "
                      f"loss {out['loss'].item():.4f} itc {out['loss_itc'].item():.3f} "
                      f"itm {out['loss_itm'].item():.3f} mlm {out['loss_mlm'].item():.3f} "
                      f"mim {out['loss_mim'].item():.3f}")
        if rank == 0:
            torch.save({"model": model.module.state_dict(), "config": cfg},
                       os.path.join(cfg["output_dir"], f"uit_epoch{epoch}.pth"))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
