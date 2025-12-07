# 🎭 Emotion & Sentiment Analysis

A Natural Language Processing project that performs **emotion classification** using both traditional machine learning and deep learning models.

The system predicts one of the following emotions:

- 😢 sadness  
- 😊 joy  
- ❤️ love  
- 😡 anger  
- 😨 fear  
- 😲 surprise  

It also maps these emotions into sentiment categories:

| Emotion              | Sentiment    |
|----------------------|--------------|
| joy, love            | **positive** |
| sadness, anger, fear | **negative** |
| surprise             | **neutral**  |
---

## 📌 Overview

This project explores several emotion-classification methods:

### 🔹 Traditional Machine Learning
- Logistic Regression (TF-IDF)
- Multinomial Naive Bayes (Bag-of-Words)

### 🔹 Deep Learning
- BiLSTM neural network (PyTorch)

Preprocessing utilities, vectorizers, and trained models are included.

---

## 🧠 Workflow Diagram

```
               Raw Text
                   |
                   v
        -------------------------
        Text Preprocessing
        (cleaning, tokenizing)
        -------------------------
         |        |          |
         v        v          v
      Bag of     TF-IDF   Token Indexing
       Words
         |          |          |
         v          v          v
     Naive Bayes   Logistic   BiLSTM
                  Regression
         |          |           |
        ------------+------------
                    |
                    v
              Emotion Prediction
```

---

## ✨ Features

- 🎯 Predicts emotions directly from text  
- 🔄 Multiple model types for experimentation  
- 🧹 Preprocessing and text-cleaning utilities  
- 💾 Saved models and vectorizers ready for use
- 📊 Visual comparison of model confidence
- 🚀 Easily extendable for transformers  

---

## 📊 Model Performance Summary

| Model | Summary |
|-------|---------|
| Logistic Regression | Strong baseline using TF-IDF features. |
| Naive Bayes | Fast, simple, effective on sparse text. |
| BiLSTM | Best sequence understanding and strongest results. |

Detailed metrics are available in the Jupyter notebook.

---

## 🛠 Installation

Install dependencies:

```
pip install pandas numpy torch scikit-learn==1.6.1 nltk beautifulsoup4 tqdm contractions wordcloud seaborn datasets
```

---

## ▶ Example Usage

### 🔹 Logistic Regression / Naive Bayes

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

---

### 🔹Naive Bayes

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
count_vectorizer = joblib.load("../utils/count_vectorizer.pkl")

text = "I feel terrible today..."
text_clean = heavy_clean(text)
vector = count_vectorizer.transform([text_clean])

pred = nb_model.predict(vector)
emotion = id2label[int(pred[0])]

print("Naive Bayes Prediction →", emotion)

```
---

### 🔹 BiLSTM (PyTorch)

```python
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import json
import importlib
import utils.vocab_tools
importlib.reload(utils.vocab_tools)

from utils.cleaning import light_clean
from utils.vocab_tools import numericalize_tokens, pad_sequence_to_len

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

class BiLSTMEmotion(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(final_hidden)

# Load vocabulary
with open("../models/vocab.json", "r") as f:
    vocab = json.load(f)

# Load trained model
bilstm_model = BiLSTMEmotion(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_classes=6
)

bilstm_model.load_state_dict(
    torch.load("../models/bilstm_model.pth", map_location="cpu")
)
bilstm_model.eval()

def predict_bilstm(text):
    clean = light_clean(text)
    tokens = clean.split()
    nums = numericalize_tokens(tokens, vocab)
    nums = pad_sequence_to_len(nums, max_len=67)
    x = torch.tensor([nums])

    with torch.no_grad():
        logits = bilstm_model(x).softmax(dim=1)[0]
        pred = logits.argmax().item()

    return id2label[pred]

text = "I am feeling sad today"
print("BiLSTM Prediction →", predict_bilstm(text))

```

---

## 📁 Project Structure

```
Emotion-Sentiment-Analysis/
├── models/
│   ├── bilstm_model.pth
│   ├── log_reg_model.pkl
│   ├── naive_bayes_model.pkl
│   └── vocab.json
│
├── notebooks/
│   ├── model_training.ipynb
│   └── presentation.ipynb
│
├── utils/
│   ├── cleaning.py
│   ├── tokenizer.py
│   ├── vocab_tools.py
│   ├── count_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   └── heart.png
│
├── .gitignore
└── README.md

```

---

## 📦 Requirements

```
pandas
numpy
torch
scikit-learn
nltk
beautifulsoup4
tqdm
contractions
joblib
```

---

## 📚 Dataset

This project uses the **dair-ai/emotion** dataset:  
https://huggingface.co/datasets/dair-ai/emotion

---

## 🚀 Future Improvements
 
- Transformer-based models (BERT, DistilBERT, RoBERTa)  
- FastAPI backend  
- Streamlit interface  
- Improved tokenization  
- Explainability tools (LIME, SHAP)  

⭐ If you find this project useful, consider giving it a star!
