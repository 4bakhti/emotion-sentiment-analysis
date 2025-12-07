def numericalize_tokens(tokens, vocab):
    """Convert tokens to integer IDs based on the given vocabulary."""
    return [vocab.get(tok, vocab.get("<UNK>", 1)) for tok in tokens]

def pad_sequence_to_len(nums, max_len=67, pad_value=0):
    if len(nums) < max_len:
        nums = nums + [pad_value] * (max_len - len(nums))
    else:
        nums = nums[:max_len]
    return nums