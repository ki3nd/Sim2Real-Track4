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
