# 📊 Telecom Customer Churn Intelligence Platform

An end-to-end **Machine Learning churn prediction system** that helps telecom companies identify customers likely to leave and apply retention strategies.

---

## 🚀 Features

- Churn prediction using **Logistic Regression**
- **Customer Risk Scoring**
- **Retention Strategy Recommendation**
- **Customer Segmentation using K-Means**
- Interactive **Streamlit Analytics Dashboard**
- **Flask API** for real-time predictions
- Business impact estimation

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Model Training (Logistic Regression)  
5. Model Evaluation (Accuracy, F1 Score, ROC-AUC)  
6. Risk Segmentation  
7. Customer Segmentation using K-Means  

---

## 📊 Dashboard Modules

### 1️⃣ Dashboard
- Customer distribution
- Contract vs churn analysis
- Churn rate KPI
- Monthly revenue insights

### 2️⃣ Churn Prediction
Predict churn probability and risk level for a customer.

Outputs:
- Churn probability
- Risk category (Low / Medium / High)
- Recommended retention strategy
- Business impact estimation

### 3️⃣ Customer Segmentation
K-Means clustering to identify customer groups.

### 4️⃣ Risk Ranking
Identify high-risk customers for targeted retention campaigns.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Flask
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure
telecom-churn-intelligence

│

├── app.py # Streamlit dashboard

├── api.py # Flask API

├── requirements.txt

│

├── data

│ └── Telco-Customer-Churn.csv

│

├── models

│ ├── churn_model.pkl

│ └── model_features.pkl

│

└── notebook

└── churn_analysis.ipynb

---

## ▶️ Run the Project

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Run API
python api.py

### 3️⃣ Run Dashboard
streamlit run app.py


---

## 📈 Model Performance

| Metric | Score |
|------|------|
| Accuracy | ~80% |
| ROC-AUC | ~0.84 |
| Precision | Good |
| Recall | Balanced |

---

## 🎯 Business Value

This system helps telecom companies:

- Identify **high churn risk customers**
- Apply **targeted retention strategies**
- Reduce **customer acquisition cost**
- Improve **customer lifetime value**

---

## 👩‍💻 Author

**Pournima More**  
Aspiring Data Science / Machine Learning Enthusiast


