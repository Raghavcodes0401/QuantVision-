"""
QuantVision - Main CLI Entrypoint for Data Pipeline & Model Training
Orchestrates src.eda and src.models
"""

import os
from src.eda import load_and_clean_data, perform_eda
from src.models import train_models

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "market_data.csv")
    models_dir = os.path.join(base_dir, "models")

    print("=" * 60)
    print("  QuantVision: Automated EDA & Model Training Pipeline")
    print("=" * 60)

    # 1. Ingest Data
    print("\n[1/3] Ingesting and validating historical dataset...")
    df = load_and_clean_data(csv_path)
    print(f"      Total trading sessions processed: {len(df)}")

    # 2. EDA
    print("\n[2/3] Performing Exploratory Data Analysis (EDA)...")
    full_df, clean_df, summary = perform_eda(df)
    print(f"      Date Range: {summary['date_range']}")
    print(f"      52-Week Range: ₹{summary['low_52w']:,.2f} to ₹{summary['high_52w']:,.2f}")
    print(f"      Average Daily Movement: ±{summary['avg_daily_change_points']} pts (~{summary['avg_daily_change_pct']}%)")
    print(f"      Annualized Volatility: {summary['annualized_volatility_pct']}%")
    print(f"      Historical Up-Days: {summary['historical_up_pct']}%")

    # 3. Train Models
    print("\n[3/3] Training Dual-Engine ML Models...")
    metrics = train_models(clean_df, full_df, summary, models_dir)
    print(f"      Linear Regression (Price Forecast) -> R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")
    print(f"      Logistic Regression (Direction Forecast) -> Accuracy: {metrics['accuracy']:.4f}")

    print("\n[SUCCESS] All models & market context artifacts successfully updated!")
    print(f"          Artifacts written to: {models_dir}\n")

if __name__ == "__main__":
    main()
