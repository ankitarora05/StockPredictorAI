# StockPredictorAI 📈

**Predict the next 5 closing prices for any stock** using a trained Random Forest model in real time.

## 🚀 Features

- **AI‑powered forecasts** – Random Forest model trained on 10+ years of historical data.
- **Live UI** – Beautiful dark‑themed interface with interactive charts.
- **Any stock symbol** – Enter `JPM`, `AAPL`, `TSLA`, `MSFT`, `GOOGL`, or any other ticker.
- **5‑day outlook** – See predicted prices for the next 5 trading days, with percentage changes.
- **FastAPI backend** – Lightweight, high‑performance API with automatic Swagger docs.

## 🛠️ Tech Stack

- **Backend**: FastAPI, scikit‑learn, pandas, yfinance
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Model**: Random Forest Regressor (multi‑output)

## 📦 Installation

```bash
git clone https://github.com/ankitarora05/predictorAI.git
cd predictorAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 train_stock_model.py
uvicorn app:app --reload