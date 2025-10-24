# 🔍 Credit Card Fraud Detection using Unsupervised ML

Detecting fraudulent credit card transactions using **Isolation Forest** and **Autoencoder** approaches without labeled training data.

---

## 🎯 Project Overview

This project identifies fraudulent transactions in a highly imbalanced dataset  
(**492 frauds out of 284,807 transactions = 0.17%**) using two unsupervised learning techniques.


---

## 🛠️ Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras  
- **Visualization:** Matplotlib, Seaborn  

---

## 📈 Results

| Model             | Accuracy | Precision | Recall | Frauds Detected |
|-------------------|-----------|------------|---------|----------------|
| Isolation Forest  | 99.7%     | 26%        | 25%     | 125 / 492      |
| Autoencoder       | 95.0%     | 3%         | 84%     | 415 / 492      |

**🏆 Winner:** Autoencoder — caught **84% of frauds** vs **25%** for Isolation Forest.  
➡️ In fraud detection, **recall matters more than precision**.

---

## 🔍 Key Findings

### Transaction Patterns
- **Fraudulent transactions:** Higher amounts ($2K–$25K+), more spread out  
- **Normal transactions:** Mostly concentrated under $500  

### Feature Analysis
- Top differentiating PCA features: **Time, V1, V3, V2, V5**

---

## 🔬 Methodology

### Preprocessing
- StandardScaler normalization  

### Isolation Forest
- Tree-based anomaly detection  

### Autoencoder
- **Architecture:** Input(30) → Dense(16) → Dense(8) → Dense(16) → Output(30)  
- Trained only on **normal transactions**  
- **Threshold:** 95th percentile of reconstruction error  

### Evaluation
- Classification report  
- Confusion matrix analysis  
