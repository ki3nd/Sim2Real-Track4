def similarity(img_feats, txt_feats):
    """Cosine similarity (features are already L2-normalized by the model heads).
    Returns [N_txt, N_img]."""
    return txt_feats @ img_feats.t()


def topk(sim_t2i, k):
    """Top-k image indices per text query. Returns (values, indices), each [N_txt, k]."""
    return sim_t2i.topk(k=min(k, sim_t2i.size(1)), dim=1)
