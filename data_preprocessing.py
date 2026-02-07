import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def load_data():
    df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def check_stationarity(series):
    return adfuller(series)[1]

def preprocess(df, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, 3])

    return np.array(X), np.array(y), scaler
