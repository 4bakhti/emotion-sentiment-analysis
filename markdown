## ❗ Project Limitations

Although the system performs well for simple cases, several limitations affect its real-world performance and generalization:

---

### 🔹 **1. Limited Dataset Coverage**
- Dataset lacks common contractions (don’t, won’t, I’m)  
- Reduces performance on informal or natural text  

---

### 🔹 **2. Short and Biased Samples**
- Training examples are very short  
- Model struggles with long, complex, or conversational sentences  

---

### 🔹 **3. Only Six Emotion Categories**
Current emotions: 😊 joy • 😢 sadness • ❤️ love • 😡 anger • 😨 fear • 😮 surprise  
Missing important categories such as:
- disgust  
- confusion  
- anticipation  
- trust  

→ Emotional nuance is limited.

---

### 🔹 **4. Sentiment Labels Are Rule-Based**
- Sentiment is assigned using simple mapping rules, not learned  
- Cannot detect subtle or mixed sentiment (e.g., “happy but tired”)  

---

### 🔹 **5. Bag-of-Words Models Ignore Context**
- Logistic Regression & Naive Bayes remove word order  
Fails with:
- negation (“not happy”)  
- sarcasm  
- context-dependent meaning  

---

### 🔹 **6. BiLSTM Vocabulary Limitations**
- Uses a fixed vocabulary  
- Unseen words become `<UNK>`  
- Performance drops on slang, typos, or domain-specific terms  

---

### 🔹 **7. Aggressive Text Cleaning Removes Emotional Cues**
Removed during preprocessing:
- emojis 🙂😡😢  
- punctuation (! ? …)  
- repeated characters (“soooo happy”)  

These carry emotional meaning → removing them reduces accuracy.

---

## **Summary**
The system works well in controlled settings but struggles with:
- messy, real-world text  
- long inputs  
- subtle emotional signals  
- unseen vocabulary  
- nuanced sentiment  

Improving data quality, vocabulary handling, and model complexity would significantly improve results.
