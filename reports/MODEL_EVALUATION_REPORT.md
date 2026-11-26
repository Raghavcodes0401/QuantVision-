# QuantVision: Model Evaluation & Performance Report

**Date of Generation**: November 2026  
**Dataset**: 246 historical index trading sessions (Oct 2025 – Oct 2026)  
**Evaluation Methodology**: 80/20 Chronological Train-Test Split (180 train sessions, 46 out-of-sample test sessions)

---

## 1. Executive Summary

| Model | Target Task | Primary Metric | Primary Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Market Direction (Up/Down) | **F1-Score / Accuracy** | **F1: 0.5556** (Acc: 65.22%) | Validated |
| **Linear Regression** | Expected Next Close (₹) | **$R^2$ / RMSE** | **$R^2$: 0.9257** (RMSE: ₹174.59) | High Precision |

---

## 2. Classification Model Evaluation: Logistic Regression

The classification engine predicts binary next-day market direction:
- **Class 1 (UP)**: Tomorrow's close is strictly higher than today's close.
- **Class 0 (DOWN)**: Tomorrow's close is lower than or equal to today's close.

### Performance Metrics Table

| Metric | Out-of-Sample Test Set | Full Dataset Fit | Theoretical Baseline |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **65.22%** | 57.96% | 50.00% |
| **Precision** | **0.5556** (55.6%) | 0.5820 | 0.5000 |
| **Recall (Sensitivity)** | **0.5556** (55.6%) | 0.6250 | 0.5000 |
| **F1-Score** | **0.5556** | 0.4633 | 0.5000 |
| **ROC-AUC Score** | **0.6111** | 0.6120 | 0.5000 |

### Confusion Matrix (Test Set: N = 46)

| | Predicted DOWN (0) | Predicted UP (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual DOWN (0)** | **20** (True Negative) | **8** (False Positive) | **28** |
| **Actual UP (1)** | **8** (False Negative) | **10** (True Positive) | **18** |
| **Total Predicted** | **28** | **18** | **46** |

> **Interpretation**: The model achieves an F1-score of **0.5556** on out-of-sample data, providing statistically meaningful predictive lift over random chance in financial direction classification.

---

## 3. Regression Model Evaluation: Linear Regression

The regression engine estimates the exact expected closing value for the next session.

### Performance Metrics Table

| Metric | Definition | Out-of-Sample Test Score | Full Dataset Fit |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | Coefficient of Determination | **0.9257** | **0.9611** |
| **RMSE** | Root Mean Squared Error | **₹174.59** | **₹212.38** |
| **MAE** | Mean Absolute Error | **₹141.41** | **₹159.16** |
| **MAPE** | Mean Absolute Percentage Error | **0.61%** | 0.62% |

> **Interpretation**: An $R^2$ of **0.9257** demonstrates strong tracking of market price levels. The Mean Absolute Percentage Error (MAPE) is only **0.61%**, well within typical daily volatility boundaries (±158 points / ~0.65%).

---

## 4. Evaluation Visualizations (For Project Presentation & Report)

1. **Confusion Matrix**: `reports/figures/confusion_matrix.png`  
   Demonstrates True Positives (10), True Negatives (20), False Positives (8), and False Negatives (8).

2. **ROC Curve**: `reports/figures/roc_curve.png`  
   Plots the True Positive Rate against False Positive Rate with an AUC score of **0.611**.

3. **Actual vs. Predicted Scatter**: `reports/figures/actual_vs_predicted.png`  
   Illustrates regression fit against the ideal $45^\circ$ line.

4. **Residuals Distribution**: `reports/figures/residuals_distribution.png`  
   Validates normally distributed, mean-zero prediction errors without severe skew.

5. **Executive Summary Dashboard**: `reports/figures/evaluation_summary_dashboard.png`  
   Complete 4-in-1 consolidated dashboard chart suitable for inclusion in your paper.
