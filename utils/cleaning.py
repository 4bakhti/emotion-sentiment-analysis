import re
import contractions
from bs4 import BeautifulSoup

def heavy_clean(text):
    """Aggressive cleaning for classical ML models."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', text)
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


import re
import contractions
from bs4 import BeautifulSoup

def light_clean(text):
    """Improved minimal cleaning for deep learning models.
    Keeps emotional content, handles emojis, negations, repeats, and punctuation."""
    
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Expand contractions: can't → cannot, i'm → i am
    text = contractions.fix(text)
    
    # Replace URLs and mentions
    text = re.sub(r'http\S+|www\.\S+', '<url>', text)
    text = re.sub(r'@\w+', '<user>', text)

    # Keep emojis (convert to words)
    emoji_map = {
        "❤️": " love ",
        "♥": " love ",
        "💕": " love ",
        "😂": " joy ",
        "😭": " cry ",
        "😢": " sad ",
        "😡": " angry ",
        "😠": " angry ",
        "😔": " sad ",
        "😞": " sad ",
        "😩": " sad ",
        "😫": " sad ",
        "😡": " anger ",
    }
    for emo, word in emoji_map.items():
        text = text.replace(emo, word)

    # Normalize repeated characters: "soooo" → "soo"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Separate punctuation from words (so LSTM sees them as tokens)
    text = re.sub(r'([!?.,])', r' \1 ', text)

    # Remove any characters that are not letters, digits, or selected punct
    text = re.sub(r"[^a-z0-9!?.,<>\s']", " ", text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def numericalize_tokens(tokens, vocab):
    unk_idx = vocab.get("<UNK>", 1)
    return [vocab.get(tok, unk_idx) for tok in tokens]