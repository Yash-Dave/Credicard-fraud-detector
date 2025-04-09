# 💳 Credit Card Fraud Detection System

An intelligent system that detects **anomalous or fraudulent credit card transactions** using **unsupervised machine learning** — all packed into an interactive, user-friendly **Streamlit dashboard**.

---

## 🧠 What This Project Does

✨ Detects potential fraud in high-volume transactional data  
🧭 Automatically identifies timestamp, amount, merchant, and other relevant columns  
🧪 Supports real and synthetic datasets  
📈 Visualizes anomalies and evaluates model performance  
🖼️ Simple, browser-based UI with real-time feedback

---

## 🔍 Techniques Used

| Model              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| ✅ Isolation Forest  | Randomly isolates anomalies based on partitioning                          |
| ✅ One-Class SVM     | Learns the decision boundary of normal transactions                        |
| ✅ Local Outlier Factor | Compares local density deviations                                         |
| ✅ Autoencoder        | Neural network that reconstructs input, flags large reconstruction errors |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Yash-Dave/Credicard-fraud-detector.git
cd credit-card-fraud-detection

```
### 2. Create & Activate Virtual Environment

``` bash

python -m venv env
source env/bin/activate  # macOS/Linux
# OR
env\Scripts\activate     # Windows

```
### 3.  Install Dependencies

```bash

pip install -r requirements.txt

```

### 4. Launch the App

``` bash
streamlit run app.py
```

## 🔮 Future Improvements

Add SHAP / LIME explainability support 🧩

Real-time data ingestion (Kafka, RabbitMQ) 🌐

Deploy as REST API with authentication 🔐

Retraining pipeline with user feedback 🔁

