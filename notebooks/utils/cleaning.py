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


def light_clean(text):
    """Minimal cleaning for deep learning models."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
