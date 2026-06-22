def tokenize_caption(tokenizer, text, max_len):
    """Tokenize a raw caption the BeiT-3 way: [bos] + ids[:max_len-2] + [eos], pad to max_len.
    padding_mask: 0 for real tokens, 1 for padding."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_len - 2:
        ids = ids[:max_len - 2]
    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    num_tokens = len(ids)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    ids = ids + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return ids, padding_mask
