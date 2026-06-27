"""CMP Stage-1 feature extraction + ITM scoring + text embedding (ported from SSDC)."""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ruamel.yaml import YAML
from transformers import BertTokenizer

from models.model_search import Search

yaml = YAML(typ="safe")


def load_cmp_components(config_path, checkpoint_path, device):
    config = yaml.load(open(config_path, 'r'))
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return config, tokenizer, model


@torch.no_grad()
def extract_cmp_features(model, data_loader, tokenizer, device, config):
    model.eval()
    texts = data_loader.dataset.text
    text_bs = config.get('batch_size_test_text', 256)
    text_feats_list, text_embeds_list, text_atts_list = [], [], []

    for i in tqdm(range(0, len(texts), text_bs), desc="Text Features"):
        batch_text = texts[i:i + text_bs]
        text_input = tokenizer(
            batch_text, padding='max_length', truncation=True,
            max_length=config['max_tokens'], return_tensors="pt",
        ).to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)
        text_embeds_list.append(text_embed.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        text_feats_list.append(F.normalize(text_feat, dim=-1).cpu())

    image_feats_list, image_embeds_list, img_paths = [], [], []
    for image, pose, img_idx in tqdm(data_loader, desc="Image Features"):
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        if config.get('be_pose_img', False) and pose is not None:
            pose = pose.to(device)
            if model.be_pose_conv:
                pose = model.pose_conv(pose)
            pose_embed, _ = model.get_vision_embeds(pose)
            image_embed = model.pose_block(image_embed, pose_embed)
        image_feat = model.get_image_feat(image_embed)
        image_embeds_list.append(image_embed.cpu())
        image_feats_list.append(F.normalize(image_feat, dim=-1).cpu())
        for idx in img_idx:
            path = data_loader.dataset.ann[idx.item()]['image']
            img_paths.append(os.path.join(data_loader.dataset.image_root, path))

    return {
        'sims_itc': torch.cat(text_feats_list, dim=0) @ torch.cat(image_feats_list, dim=0).t(),
        'text_feats': torch.cat(text_feats_list, dim=0),
        'image_feats': torch.cat(image_feats_list, dim=0),
        'text_embeds': torch.cat(text_embeds_list, dim=0),
        'image_embeds': torch.cat(image_embeds_list, dim=0),
        'text_atts': torch.cat(text_atts_list, dim=0),
        'img_paths': img_paths,
        'texts': texts,
    }


@torch.no_grad()
def compute_cmp_itm_scores(model, features, device, config):
    sims_matrix = features['sims_itc'].to(device)
    image_embeds = features['image_embeds'].to(device)
    text_embeds = features['text_embeds'].to(device)
    text_atts = features['text_atts'].to(device)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0, device=device)
    k_test = config.get('k_test', 128)

    for i, sims in enumerate(tqdm(sims_matrix, desc="ITM Re-ranking")):
        topk_sim, topk_idx = sims.topk(k=min(k_test, sims.size(0)), dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        current_text_embed = text_embeds[i].unsqueeze(0).repeat(encoder_output.size(0), 1, 1)
        current_text_att = text_atts[i].unsqueeze(0).repeat(encoder_output.size(0), 1)
        output = model.get_cross_embeds(
            encoder_output, encoder_att,
            text_embeds=current_text_embed, text_atts=current_text_att,
        )[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    mask = score_matrix_t2i == 1000.0
    score_matrix_t2i[mask] = replacement[mask]

    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_sim = (sims_matrix - sims_matrix.min()) / (sims_matrix.max() - sims_matrix.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim
    return score_matrix_t2i.cpu()


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, config):
    """L2-normalized CMP text features for an arbitrary list of strings (T or T_new)."""
    max_len = config.get('max_tokens_text', config.get('max_tokens', 56))
    text_input = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_len, return_tensors="pt",
    ).to(device)
    text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_feat = model.get_text_feat(text_embed)
    return F.normalize(text_feat, dim=-1)
