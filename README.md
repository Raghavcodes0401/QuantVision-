# QuantVision 📈

**QuantVision** is an executive quantitative market forecasting web application. By entering the last day's closing price, the app analyzes 1-year historical trends and delivers clean, actionable predictions without cluttering the screen with complex diagnostic graphs.

---

## Key Features

- **Directional Prediction (Up / Down)**: Powered by **Logistic Regression**, predicting next-session market momentum and confidence probability.
- **Expected Closing Price**: Powered by **Linear Regression**, calculating target price, point change, and percentage movement.
- **Abstracted Context**: User-friendly market vital signs including 52-week position, average daily movement, and volatility.
- **Minimalist Forecast Chart**: Clean Chart.js visual showing recent historical price action leading into tomorrow's projected close.

---

## Project Structure

```
quantvision/
├── app.py                     # Flask web server and prediction API
├── eda_and_train.py           # Automated EDA and model training pipeline
├── data/
│   └── market_data.csv        # Historical index data (Oct 2025 – Oct 2026)
├── models/
│   ├── linear_model.pkl       # Trained Linear Regression model
│   ├── logistic_model.pkl     # Trained Logistic Regression model
│   ├── scaler.pkl             # Feature scaler
│   └── market_context.json    # Summary metrics & recent price sequence
├── templates/
│   └── index.html             # Dashboard UI (Tailwind CSS + Chart.js)
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore file
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/QuantVision.git
cd QuantVision
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.
