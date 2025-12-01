def numericalize_tokens(tokens, vocab):
    """Convert tokens to integer IDs based on the given vocabulary."""
    return [vocab.get(tok, vocab.get("<UNK>", 1)) for tok in tokens]
