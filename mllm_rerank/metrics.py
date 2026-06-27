"""Retrieval metrics (R@K / mAP / mINP) — ported from SSDC vllm_infer_SSDC.py."""
import torch
from prettytable import PrettyTable


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.)
           for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    row = [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
           t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]
    if retur_indices:
        return row, indices
    return row


def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
    for key in sims_dict.keys():
        rs = get_metrics(sims_dict[key], qids, pids, f'{key}-t2i', False)
        table.add_row(rs)
    for col in ["R1", "R5", "R10", "mAP", "mINP", "rSum"]:
        table.custom_format[col] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))
