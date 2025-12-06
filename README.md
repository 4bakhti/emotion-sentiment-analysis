# 🎭 Emotion & Sentiment Analysis

A Natural Language Processing project that performs **emotion classification** using both traditional machine learning and deep learning models.

The system predicts one of the following emotions:

- 😢 sadness  
- 😊 joy  
- ❤️ love  
- 😡 anger  
- 😨 fear  
- 😲 surprise  

Future update: sentiment classification (positive / neutral / negative).

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
           |        |        |
           v        v        v
      Bag of     TF-IDF   Token Indexing
       Words
           |        |        |
           v        v        v
     Naive Bayes   Logistic   BiLSTM
                    Regression
           |         |         |
           -----------+---------
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
- 🚀 Easily extendable for deep learning and transformers  

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
pip install pandas numpy torch scikit-learn nltk beautifulsoup4 tqdm contractions joblib
```

---

## ▶ Example Usage

### 🔹 Logistic Regression / Naive Bayes

```python
import joblib
from utils.cleaning import heavy_clean

model = joblib.load("models/log_reg_model.pkl")
vectorizer = joblib.load("utils/tfidf_vectorizer.pkl")

text = "I am very happy today!"
clean = heavy_clean(text)
vector = vectorizer.transform([clean])

prediction = model.predict(vector)
print(prediction)  # Example: 1 (joy)
```

---

### 🔹 BiLSTM (PyTorch)

```python
import torch
import json
from utils.cleaning import light_clean, numericalize_tokens
from models.bilstm_model import BiLSTMEmotion

with open("models/vocab.json", "r") as f:
    vocab = json.load(f)

model = BiLSTMEmotion(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_classes=6
)

model.load_state_dict(torch.load("models/bilstm_model.pth"))
model.eval()

text = "I am very happy today!"
tokens = light_clean(text).split()
numericalized = numericalize_tokens(tokens, vocab)
tensor_input = torch.tensor([numericalized])

with torch.no_grad():
    output = model(tensor_input)
    prediction = torch.argmax(output).item()

print(prediction)
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
├── notebooks/
│   └── emotion_sentiment_analysis.ipynb
├── utils/
│   ├── cleaning.py
│   ├── tokenizer.py
│   ├── vocab_tools.py
│   ├── count_vectorizer.pkl
│   └── tfidf_vectorizer.pkl
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

- Sentiment classification  
- Transformer-based models (BERT, DistilBERT, RoBERTa)  
- FastAPI backend  
- Streamlit interface  
- Improved tokenization  
- Explainability tools (LIME, SHAP)  

