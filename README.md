# 🧠 Classify Transactions

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
</p>

## 🧩 Part of the fin-track Ecosystem
This repository contains the Machine Learning logic and data science workflows used to categorize financial transactions for the **[Finance Tracker](https://github.com/your-username/fin-track)** project. 

## ✨ Overview
The model takes raw bank statement strings (extracted via OCR) and classifies them into structured financial categories (e.g., Food, Transport, Housing) using NLP techniques and classification algorithms.

## 📂 Project Structure
```text
Classify/
├── api/                  # Serverless function handlers (Vercel)
│   ├── classify.py       # Main inference endpoint
│   ├── _utils.py         # Utility functions for the API
│   └── classify_model.pkl# Serialized Machine Learning model
├── data/                 # Raw data and test files (ignored in git)
├── notebooks/            # Jupyter notebooks for data exploration
├── scripts/              # MLOps and utility scripts
│   └── train_model.py    # Script to retrain the ML model
├── README.md             
└── requirements.txt      
```

## 🚀 How to Explore & Run

### 1. Setup Environment
```bash
git clone https://github.com/your-username/classify_transactions.git
cd classify_transactions
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. MLOps (Retraining the Model)
To generate new data and retrain the Random Forest model:
```bash
python scripts/train_model.py
```
This will automatically save the new model to `api/classify_model.pkl`.

### 3. API Deployment (Vercel)
This project is built using Python's `http.server.BaseHTTPRequestHandler` to be easily deployed as Serverless Functions on Vercel. 
To test the API locally, run the test server:
```bash
python -m api._test
```
You can then send POST requests with transaction descriptions to classify them.
