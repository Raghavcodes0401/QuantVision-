"""
QuantVision - Web Application & Prediction Server
Serves the executive market forecasting dashboard and JSON prediction API.
"""

import os
from flask import Flask, render_template, request, jsonify
from src.predictor import MarketPredictor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Initialize prediction engine
predictor = MarketPredictor(MODELS_DIR)

@app.route('/')
def index():
    default_close = predictor.eda_summary['latest_close']
    initial_prediction = predictor.predict(default_close)
    return render_template('index.html', initial=initial_prediction, summary=predictor.eda_summary)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    try:
        input_close = float(data.get('close', predictor.eda_summary['latest_close']))
        result = predictor.predict(input_close)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"QuantVision server active on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
