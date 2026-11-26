"""
Model training module for QuantVision.
Trains:
  1. Linear Regression for expected next-day close
  2. Logistic Regression for market direction probability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

FEATURE_NAMES = ['Close', 'Daily_Return', 'Ratio_SMA5', 'Ratio_SMA20', 'Vol_5']

def train_models(clean_df: pd.DataFrame, df_full: pd.DataFrame, summary: dict, output_dir: str):
    """Train both models, evaluate metrics, and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    X = clean_df[FEATURE_NAMES].values
    y_reg = clean_df['Next_Close'].values
    y_clf = clean_df['Market_Up'].values

    # 1. Linear Regression for Expected Next-Day Close
    lin_reg = LinearRegression()
    lin_reg.fit(X, y_reg)
    preds_reg = lin_reg.predict(X)
    r2 = r2_score(y_reg, preds_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, preds_reg))

    # 2. Logistic Regression for Direction (UP / DOWN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_scaled, y_clf)
    preds_clf = log_reg.predict(X_scaled)
    acc = accuracy_score(y_clf, preds_clf)

    # Save trained objects
    joblib.dump(lin_reg, os.path.join(output_dir, "linear_model.pkl"))
    joblib.dump(log_reg, os.path.join(output_dir, "logistic_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # Recent history sequence for inference feature construction
    recent_tail = df_full[['Date', 'Close', 'SMA_5', 'SMA_20', 'Daily_Return']].tail(25).copy()
    recent_tail['Date'] = recent_tail['Date'].dt.strftime('%d-%b-%Y')
    recent_records = recent_tail.to_dict(orient='records')

    market_context = {
        "eda_summary": summary,
        "features": FEATURE_NAMES,
        "recent_history": recent_records,
        "model_stats_abstracted": {
            "samples": len(clean_df),
            "linear_r2": float(r2),
            "linear_rmse": float(rmse),
            "logistic_acc": float(acc)
        }
    }

    with open(os.path.join(output_dir, "market_context.json"), "w") as f:
        json.dump(market_context, f, indent=2)

    return {
        "r2": r2,
        "rmse": rmse,
        "accuracy": acc
    }
