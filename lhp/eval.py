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
