# Network Attack & Anomaly Detection System

## About the Project

This project is a Machine Learning based Network Attack and Anomaly Detection System. It analyzes network traffic data and predicts whether the traffic is normal, anomalous, or part of a cyber attack.

The system uses trained machine learning models and preprocessing techniques to identify suspicious activities in real time.

---

## Features

- Detects network anomalies
- Identifies possible cyber attacks
- Machine Learning based prediction
- Simple web interface
- Fast and easy deployment

---

## Project Structure

├── app.js                 # Frontend functionality

├── style.css              # UI styling

├── capture.py             # Network packet capture script

├── anomaly_xgb.pkl        # Trained anomaly detection model

├── le_attack.pkl          # Label encoder for attack classes

├── feature_cols.pkl       # Feature column information

├── scaler_anomaly.pkl     # Scaler for anomaly detection

├── scaler_attack.pkl      # Scaler for attack detection

├── requirements.txt       # Required Python packages

└── README.md

---

## Technologies Used

- Python
- Machine Learning
- XGBoost
- HTML
- CSS
- JavaScript
- Scikit-learn
  

---

## Installation

1. Clone the repository

```bash
git clone <repository-url>
