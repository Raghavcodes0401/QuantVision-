"""
Exploratory Data Analysis (EDA) module for QuantVision.
Performs data cleaning, chronological sorting, technical feature computation,
and extracts high-level user-friendly market summaries.
"""

import numpy as np
import pandas as pd

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load historical market data, clean column names, and sort chronologically."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date').reset_index(drop=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (₹ Cr)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def perform_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute daily returns, moving averages, rolling volatility, targets,
    and build an abstracted user-friendly EDA summary.
    Returns:
      - full_df: Dataframe with all computed columns
      - clean_df: Dropped NaN rows ready for ML training
      - summary: High-level user-friendly EDA metrics
    """
    data = df.copy()

    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Change'] = data['Close'].diff()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Vol_5'] = data['Daily_Return'].rolling(window=5).std()

    # Targets:
    # 1. Continuous price target for Linear Regression
    data['Next_Close'] = data['Close'].shift(-1)
    # 2. Binary market direction target for Logistic Regression (1 = UP, 0 = DOWN)
    data['Market_Up'] = (data['Next_Close'] > data['Close']).astype(int)

    # Ratios
    data['Ratio_SMA5'] = data['Close'] / data['SMA_5']
    data['Ratio_SMA20'] = data['Close'] / data['SMA_20']

    # Filter complete feature rows for training
    clean_df = data.dropna(subset=['SMA_20', 'Vol_5', 'Next_Close']).copy().reset_index(drop=True)

    # User-friendly summary metrics
    start_date = data['Date'].min().strftime('%d %b %Y')
    end_date = data['Date'].max().strftime('%d %b %Y')
    high_52w = float(data['High'].max())
    low_52w = float(data['Low'].min())
    latest_close = float(data['Close'].iloc[-1])
    latest_date = data['Date'].iloc[-1].strftime('%d %b %Y')

    up_count = int((clean_df['Market_Up'] == 1).sum())
    up_pct = round((up_count / len(clean_df)) * 100, 1)

    avg_pts = round(float(data['Price_Change'].abs().mean()), 2)
    avg_pct = round(float(data['Daily_Return'].abs().mean() * 100), 2)
    ann_vol = round(float(data['Daily_Return'].std() * np.sqrt(252) * 100), 2)
    recent_trend = "Bullish" if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1] else "Bearish"

    summary = {
        "total_trading_days": len(data),
        "date_range": f"{start_date} to {end_date}",
        "latest_close": latest_close,
        "latest_date": latest_date,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "avg_daily_change_points": avg_pts,
        "avg_daily_change_pct": avg_pct,
        "annualized_volatility_pct": ann_vol,
        "historical_up_pct": up_pct,
        "recent_trend": recent_trend
    }

    return data, clean_df, summary
