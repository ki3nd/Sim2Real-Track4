"""CGCR agent prompts (Decomposer / Verifier / Critic / Adjudicator) + JSON parsers."""
import json
import re

DECOMPOSER_PROMPT = """You are decomposing a person-search query into atomic, independently checkable claims.
For each claim, assign a category from this set: gender, hair, upper, lower, accessory, action, anomaly, background.
Use "action" or "anomaly" for what the person is DOING (e.g., walking, falling, fighting).

Query: {query}

Return STRICTLY a JSON object: {{"claims":[{{"id":1,"category":"<cat>","text":"<short claim>"}}, ...]}}.
Keep each claim text short (a few words). Do not add commentary."""

VERIFIER_PROMPT = """You are verifying whether each claim is true of the PRIMARY person in the image.
For each claim, answer:
- "ENTAILED" if the image clearly supports it,
- "CONTRADICTED" if the image clearly shows the opposite,
- "NEUTRAL" if it cannot be determined from the image.
Give a confidence in [0,1] and one short evidence phrase.

Claims (JSON): {claims}

Return STRICTLY a JSON object: {{"verdicts":[{{"id":1,"label":"ENTAILED|NEUTRAL|CONTRADICTED","conf":0.0,"evidence":"<short>"}}, ...]}}."""

CRITIC_PROMPT = """Look again at the image and re-judge ONLY the following suspect claims.
Be strict: confirm a claim only if visually evident; otherwise mark NEUTRAL or CONTRADICTED.

Suspect claims (JSON): {claims}

Return STRICTLY a JSON object: {{"verdicts":[{{"id":1,"label":"ENTAILED|NEUTRAL|CONTRADICTED","conf":0.0,"evidence":"<short>"}}, ...]}}."""

ADJUDICATOR_PROMPT = """You are given a query and {n} candidate images (in order, index 0..{n}-1).
Rank the images from best to worst match to the query, focusing on the described ACTION/behavior first, then appearance.

Query: {query}

Return STRICTLY a JSON object: {{"ranking":[<best index>, ..., <worst index>]}} listing each index 0..{n}-1 exactly once."""


def _extract_json(text):
    """Return the first JSON object parseable from text, else None."""
    if text is None:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def parse_claims(raw):
    obj = _extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        return []
    out = []
    for i, c in enumerate(obj["claims"], start=1):
        if not isinstance(c, dict):
            continue
        out.append({
            "id": i,
            "category": str(c.get("category", "")).strip().lower(),
            "text": str(c.get("text", "")).strip(),
        })
    return out


_LABELS = {"ENTAILED", "NEUTRAL", "CONTRADICTED"}


def parse_verdicts(raw):
    obj = _extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("verdicts"), list):
        return []
    out = []
    for v in obj["verdicts"]:
        if not isinstance(v, dict) or "id" not in v:
            continue
        label = str(v.get("label", "")).strip().upper()
        if label not in _LABELS:
            label = "NEUTRAL"
        try:
            conf = float(v.get("conf", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append({
            "id": int(v["id"]),
            "label": label,
            "conf": conf,
            "evidence": str(v.get("evidence", "")).strip(),
        })
    return out


def parse_adjudication(raw, n):
    identity = list(range(n))
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return identity
    rank = obj.get("ranking")
    if not isinstance(rank, list) or len(rank) != n:
        return identity
    try:
        rank = [int(x) for x in rank]
    except Exception:
        return identity
    if sorted(rank) != identity:   # must be a permutation of 0..n-1
        return identity
    return rank
