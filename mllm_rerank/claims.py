"""Claim weighting, self-consistency aggregation, and the grounded entailment score."""
from collections import defaultdict


def category_weight(category, table, default):
    return table.get(str(category).strip().lower(), default)


def aggregate_self_consistency(samples):
    """samples: list of verdict-lists (each verdict {id,label,conf}). Majority label per id;
    conf = mean conf of majority-label verdicts; label-count tie -> NEUTRAL."""
    by_id = defaultdict(list)
    for sample in samples:
        for v in sample:
            by_id[v["id"]].append(v)

    out = []
    for vid in sorted(by_id.keys()):
        verds = by_id[vid]
        counts = defaultdict(int)
        for v in verds:
            counts[v["label"]] += 1
        top = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == top]
        if len(winners) > 1:
            label = "NEUTRAL"
        else:
            label = winners[0]
        confs = [v["conf"] for v in verds if v["label"] == label]
        conf = sum(confs) / len(confs) if confs else 0.0
        out.append({"id": vid, "label": label, "conf": conf})
    return out


def score_ground(claims, verdicts, table, default):
    vmap = {v["id"]: v for v in verdicts}
    num = 0.0
    den = 0.0
    for c in claims:
        w = category_weight(c.get("category", ""), table, default)
        den += w
        v = vmap.get(c["id"])
        if v is None:
            continue
        if v["label"] == "ENTAILED":
            num += w * v["conf"]
        elif v["label"] == "CONTRADICTED":
            num += w * (-v["conf"])
        # NEUTRAL contributes 0
    if den == 0.0:
        return 0.0
    return num / den
