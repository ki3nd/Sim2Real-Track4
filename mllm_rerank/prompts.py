"""Detective / Analyst / Writer prompts and output parsers (ported from SSDC)."""
import json
import re

DETECTIVE_PROMPT = """You are an expert in Person Re-identification and Anomaly Detection.
Task: Determine if the text accurately describes the **primary person** in the image.

Evaluation Criteria:
1.**Appearance**: Check gender, clothing color, clothing type (upper/lower), and distinct accessories.
2.**Action**: Check if the described action (e.g., walking, falling, fighting) matches the person's behavior.
3.**Ignore**: Do not judge based on background details or lighting differences unless they obscure the person.

Text: {cap}

Does this text accurately describe the image? Answer STRICTLY with "Yes" or "No"."""

ANALYST_PROMPT = """According to the pedestrian image, answer the following questions one by one:

1.The person is male or female?
2.What hairstyle does the person have, such as hair length and color?
3.What is this person wearing on his upper body? If clearly visible, what are the color, type, and sleeve length?
4.What are the characteristics of this person's pants? If clearly visible, what are the color, type, and trouser leg length?
5.Does this person have any patterns on his/her clothes or pants?
6.What are the characteristics of this person's shoes? If clearly visible, what are the color and style?
7.Does this person wear glasses? If clearly visible, what are the color and style?
8.Is this person wearing a scarf? If clearly visible, what are the color and style?
9.Does this person have something in his/her hand? If so, what is it and what color is it?
10.Does this person carry a backpack? If clearly visible, what are the color and style?
11.Does this person wear a hat? If clearly visible, what are the color and style?
12.Is this person wearing a belt or waistband?
13.What is this person doing?
14.What is the background?
15.Are there other people in the background of this person?"""

WRITER_PROMPT = """
    Task: Aggregate the following subtexts into a single continuous and concise text paragraph.
    Format: Return the result strictly as a JSON object with a single key "caption".

    Requirements:

    1.Grammar Flow: Ensure the transition after the prefix is natural.
    2.Keep the output concise, fluent and grammatical.
    3.The final returned content must be a JSON object with the single key "caption" whose value is the aggregated caption string.

    Now let's get started.
    Subtexts: {cap}
    Output:
    """


def process_checklist(raw_answers):
    """Clean raw Analyst answers into per-sample lists of sentences (ports SSDC process_cap_)."""
    tmps = []
    for c in raw_answers:
        c = c.split('\n')
        tmp = []
        for cc in c:
            if ': ' in cc:
                continue
            try:
                cc = cc.split('.')[1]
                if 'Yes, ' in cc:
                    cc = cc.replace('Yes, ', '')
                if 'No, ' in cc:
                    cc = cc.replace('No, ', '')
                cc = cc[:1].upper() + cc[1:]
                if cc[-1:] != '.':
                    cc += '.'
            except Exception:
                cc = ''
            tmp.append(cc)
        tmps.append(tmp)
    return tmps


def parse_writer_caption(raw):
    """Extract the "caption" value from the Writer's JSON; fall back to stripped raw text."""
    if raw is None:
        return ""
    text = raw.strip()
    # try direct JSON parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "caption" in obj:
            return str(obj["caption"]).strip()
    except Exception:
        pass
    # try to find an embedded {...} block containing "caption"
    match = re.search(r'\{.*?"caption".*?\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "caption" in obj:
                return str(obj["caption"]).strip()
        except Exception:
            pass
    return text
