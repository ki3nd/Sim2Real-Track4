import os
import argparse
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ruamel.yaml import YAML
from transformers import XLMRobertaTokenizer

from lhp.transform import LHPTransform
from lhp.dataset import LHPDataset
from lhp.model import LHPRetriever
from lhp.tokenization import tokenize_caption
from lhp.masked_model import build_vision_padding_mask


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lhp/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = YAML(typ="safe").load(f)

    local_rank, rank, world = setup_ddp()          # BEFORE building model (ClipLoss rank/world)
    device = torch.device("cuda", local_rank)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tokenizer = XLMRobertaTokenizer(cfg["spm_model"])
    transform = LHPTransform(cfg["resolution"], tuple(cfg["crop_scale"]),
                             cfg["local_prob"], cfg.get("masked_prob", 0.0))
    data_root = cfg["data_root"]
    ann_files = [os.path.join(data_root, f) for f in cfg["train_file"]]
    dataset = LHPDataset(ann_files, data_root, transform,
                         max_words=cfg["max_words"], eda=cfg["eda"], eda_p=cfg["eda_p"])

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    per_gpu_bs = cfg["batch_size"] // world
    loader = DataLoader(dataset, batch_size=per_gpu_bs, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = LHPRetriever(ckpt_path=cfg["beit3_ckpt"], drop_path_rate=cfg["drop_path_rate"]).to(device)
    # BeiT-3's vision_embed.mask_token is unused in the retrieval forward (it is a
    # masked-image-pretraining param), so plain DDP would error on its missing grad.
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = math.ceil(len(dataset) / cfg["batch_size"]) * cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        for i, (image, captions, _path, patch_mask) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            vision_padding_mask = build_vision_padding_mask(patch_mask.to(device))  # [B, 1+num_patches]
            toks = [tokenize_caption(tokenizer, c, cfg["max_tokens"]) for c in captions]
            text_ids = torch.tensor([t[0] for t in toks], device=device)
            padding_mask = torch.tensor([t[1] for t in toks], device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(image, text_ids, padding_mask, vision_padding_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if rank == 0 and i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(loader)} loss {loss.item():.4f}")

        if rank == 0:
            torch.save({"model": model.module.state_dict(), "config": cfg},
                       os.path.join(cfg["output_dir"], f"lhp_epoch{epoch}.pth"))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
