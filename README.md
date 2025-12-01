# Emotion & Sentiment Analysis

## Project Overview

This project classifies text into six emotions:

* sadness
* joy
* love
* anger
* fear
* surprise

**Note:** Sentiment mapping (positive, negative, neutral) will be added in future updates.

The project implements three models for prediction:

1. Logistic Regression (TF-IDF)
2. Multinomial Naive Bayes (Bag-of-Words)
3. BiLSTM (PyTorch)

Preprocessing and cleaning utilities are included, and trained models with vectorizers are provided for easy inference.

---

## Features

* Predict emotions from text.
* Multiple model options for experimentation.
* Preprocessing utilities included.
* Ready-to-use saved models and vectorizers.
* Sentiment prediction will be added later.

---

## Installation

Install required packages:

```bash
pip install pandas numpy torch scikit-learn nltk beautifulsoup4 tqdm contractions
```

---

## Usage Example

### Logistic Regression / Naive Bayes

```python
import joblib
from utils.cleaning import heavy_clean

# Load saved Logistic Regression model
log_reg_model = joblib.load("models/log_reg_model.pkl")

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load("utils/tfidf_vectorizer.pkl")

# Example prediction
text = "I am very happy today!"
text_clean = heavy_clean(text)
vector = tfidf_vectorizer.transform([text_clean])
pred = log_reg_model.predict(vector)
print(pred)  # Output: 1 (joy)
```

### BiLSTM (PyTorch)

```python
import torch
from utils.cleaning import light_clean, numericalize_tokens
from models.bilstm_model import BiLSTMEmotion  # if you put class in a separate file
import json

# Load vocabulary
with open("models/vocab.json", "r") as f:
    vocab = json.load(f)

# Initialize model
model_BiLSTM = BiLSTMEmotion(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, num_classes=6)
model_BiLSTM.load_state_dict(torch.load("models/bilstm_model.pth"))
model_BiLSTM.eval()

# Example prediction
text = "I am very happy today!"
text_clean = light_clean(text)
text_numerical = torch.tensor([numericalize_tokens(text_clean.split(), vocab)])
with torch.no_grad():
    output = model_BiLSTM(text_numerical)
    pred = torch.argmax(output, dim=1).item()
print(pred)  # Output: 1 (joy)
```

---

## Folder Structure

```
.
├── models/                           # Saved trained models and vocabulary artifacts
│   ├── bilstm_model.pth              # PyTorch Bi-LSTM model weights
│   ├── log_reg_model.pkl             # Trained Logistic Regression model
│   ├── naive_bayes_model.pkl         # Trained Naive Bayes model
│   └── vocab.json                    # Vocabulary mapping for the deep learning model
├── notebooks/                        # Jupyter notebooks for exploration and training
│   └── emotion_sentiment_analysis.ipynb  # Main notebook for analysis, training, and testing
├── utils/                            # Helper scripts and pre-fitted vectorizers
│   ├── cleaning.py                   # Functions for text cleaning and preprocessing
│   ├── count_vectorizer.pkl          # Saved CountVectorizer object
│   ├── tfidf_vectorizer.pkl          # Saved TfidfVectorizer object
│   ├── tokenizer.py                  # Custom tokenization logic
│   └── vocab_tools.py                # Utilities for vocabulary management
└── README.md                         # Project documentation and setup instructions
```

---

## Dataset

The project uses the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset

