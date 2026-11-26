"""
Prediction and inference engine for QuantVision.
Loads saved models and calculates the next-day price & direction
when a user enters the last day closing price.
"""

import os
import json
import joblib
import numpy as np

class MarketPredictor:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.lin_reg = joblib.load(os.path.join(models_dir, "linear_model.pkl"))
        self.log_reg = joblib.load(os.path.join(models_dir, "logistic_model.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

        with open(os.path.join(models_dir, "market_context.json"), "r") as f:
            self.context = json.load(f)

        self.eda_summary = self.context['eda_summary']
        self.recent_history = self.context['recent_history']

    def predict(self, input_close: float) -> dict:
        """
        Calculates expected next-day close and market direction
        given the user's input last closing price.
        """
        prev_closes = [r['Close'] for r in self.recent_history[-20:]]
        p_last = prev_closes[-1]

        # Feature computation
        daily_return = (input_close - p_last) / p_last if p_last != 0 else 0.0
        sma5 = np.mean(prev_closes[-4:] + [input_close])
        sma20 = np.mean(prev_closes[-19:] + [input_close])
        r_sma5 = input_close / sma5 if sma5 != 0 else 1.0
        r_sma20 = input_close / sma20 if sma20 != 0 else 1.0

        recent_5_closes = prev_closes[-4:] + [input_close]
        rets = [(b - a) / a for a, b in zip(recent_5_closes[:-1], recent_5_closes[1:])]
        vol5 = float(np.std(rets)) if len(rets) > 1 else 0.008

        X = np.array([[input_close, daily_return, r_sma5, r_sma20, vol5]])

        # 1. Linear Regression -> Expected Close
        predicted_close = float(self.lin_reg.predict(X)[0])
        change_pts = predicted_close - input_close
        change_pct = (change_pts / input_close) * 100 if input_close != 0 else 0.0

        # 2. Logistic Regression -> Market Direction (UP / DOWN)
        X_scaled = self.scaler.transform(X)
        prob_up = float(self.log_reg.predict_proba(X_scaled)[0][1])
        prob_down = 1.0 - prob_up

        is_up = prob_up >= 0.5
        direction = "UP" if is_up else "DOWN"
        direction_label = "BULLISH (UP)" if is_up else "BEARISH (DOWN)"
        confidence = round((prob_up if is_up else prob_down) * 100, 1)

        # Expected Corridor
        avg_mov = self.eda_summary['avg_daily_change_points']
        corridor_low = round(predicted_close - (avg_mov * 0.5), 2)
        corridor_high = round(predicted_close + (avg_mov * 0.5), 2)

        # 52-week position
        high_52 = self.eda_summary['high_52w']
        low_52 = self.eda_summary['low_52w']
        pos_52 = round(((input_close - low_52) / (high_52 - low_52)) * 100, 1) if high_52 > low_52 else 50.0
        pos_52 = max(0.0, min(100.0, pos_52))

        # Chart trajectory
        chart_dates = [r['Date'] for r in self.recent_history[-15:]]
        chart_prices = [round(r['Close'], 2) for r in self.recent_history[-15:]]

        chart_dates.append("Input Close")
        chart_prices.append(round(input_close, 2))

        chart_dates.append("Next Session (Forecast)")
        chart_prices.append(round(predicted_close, 2))

        return {
            "input_close": round(input_close, 2),
            "predicted_close": round(predicted_close, 2),
            "change_pts": round(change_pts, 2),
            "change_pct": round(change_pct, 2),
            "direction": direction,
            "direction_label": direction_label,
            "confidence": confidence,
            "corridor_low": corridor_low,
            "corridor_high": corridor_high,
            "pos_52": pos_52,
            "chart_dates": chart_dates,
            "chart_prices": chart_prices,
            "eda_summary": self.eda_summary
        }
