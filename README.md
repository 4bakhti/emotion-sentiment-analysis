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
pip install pandas numpy torch scikit-learn==1.6.1 nltk beautifulsoup4 tqdm contractions
```

---

## Usage Example

### Logistic Regression

```python
import sys
sys.path.append("..")
import joblib
from utils.cleaning import heavy_clean

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load Logistic Regression model
log_reg_model = joblib.load("../models/log_reg_model.pkl")
tfidf_vectorizer = joblib.load("../utils/tfidf_vectorizer.pkl")

# Example prediction
text = "I am very happy today!"
text_clean = heavy_clean(text)
vector = tfidf_vectorizer.transform([text_clean])
pred = log_reg_model.predict(vector)
emotion = id2label[int(pred[0])]
print("Logistic Regression Prediction →", emotion)

```

### Naive Bayes
```python
import sys
sys.path.append("..")

import joblib
from utils.cleaning import heavy_clean

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load Naive Bayes model
nb_model = joblib.load("../models/naive_bayes_model.pkl")

# Load CountVectorizer
count_vectorizer = joblib.load("../utils/count_vectorizer.pkl")

# Example prediction
text = "I feel terrible today..."
text_clean = heavy_clean(text)
vector = count_vectorizer.transform([text_clean])

pred = nb_model.predict(vector)
emotion = id2label[int(pred[0])]
print("Naive Bayes model Prediction →",emotion) 

```

### BiLSTM (PyTorch)

```python
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import json

from utils.cleaning import light_clean
from utils.vocab_tools import numericalize_tokens

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


# BiLSTM Model Definition
class BiLSTMEmotion(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(final_hidden)



#Vocabulary
with open("../models/vocab.json", "r") as f:
    vocab = json.load(f)

#Load Trained BiLSTM Model
bilstm_model = BiLSTMEmotion(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_classes=6
)

bilstm_model.load_state_dict(torch.load("../models/bilstm_model.pth", map_location="cpu"))
bilstm_model.eval()



def predict_bilstm(text):
    clean = light_clean(text)
    tokens = clean.split()
    nums = numericalize_tokens(tokens, vocab)
    x = torch.tensor([nums])

    with torch.no_grad():
        logits = bilstm_model(x)
        pred = torch.argmax(logits, dim=1).item()

    return id2label[pred]

# Example Prediction
text = "I am extremely happy today!"
print("BiLSTM Prediction →", predict_bilstm(text))
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

