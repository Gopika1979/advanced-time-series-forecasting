# Advanced Time Series Forecasting with Neural Networks and Explainability

## Project Overview
This project implements a production-ready deep learning model for multivariate time series forecasting.  
It uses an **LSTM network** to predict future values of a complex dataset and applies **SHAP** for model interpretability.

---

## Dataset
- Source: Publicly available dataset (e.g., financial market data)
- Features used (minimum 5):
  - Open
  - High
  - Low
  - Close
  - Volume
- Preprocessing steps:
  - Normalization
  - Sequence generation for LSTM input
  - Stationarity checks

---

## Model Implementation
- **Model**: LSTM
- **Hyperparameter tuning**: Bayesian optimization / Keras Tuner
- **Evaluation metrics**: RMSE, MAE, MAPE
- **Baseline comparison**: SARIMAX

**Results**:
- RMSE: 0.0387  
- MAE: 0.0332  
- MAPE: 4.01%

---

## Model Interpretability
- **Method**: SHAP (feature importance)  
- **Top features influencing forecast**:
  | Feature | Importance |
  |---------|------------|
  | High    | 0.003682   |
  | Low     | 0.003661   |
  | Open    | 0.003650   |
  | Close   | 0.003195   |
  | Volume  | 0.000101   |

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
