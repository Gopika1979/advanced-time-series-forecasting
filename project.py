# =========================
# 1. IMPORT LIBRARIES
# =========================
import yfinance as yf
import numpy as np
import pandas as pd
import optuna
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 2. DATA ACQUISITION
# =========================
print("Downloading dataset...")
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# 3. STATIONARITY CHECK
# =========================
print("\nADF Test (Close Price):")
adf_result = adfuller(df['Close'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# =========================
# 4. NORMALIZATION
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# =========================
# 5. SEQUENCE GENERATION
# =========================
SEQ_LEN = 60

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3])  # Close price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)

print("Sequence shape:", X.shape)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 7. BASELINE MODEL (SARIMAX)
# =========================
print("\nTraining SARIMAX baseline...")
train_close = df['Close'][:split+SEQ_LEN]
test_close = df['Close'][split+SEQ_LEN:]

sarimax = SARIMAX(train_close, order=(5,1,0))
sarimax_fit = sarimax.fit(disp=False)

sarimax_preds = sarimax_fit.forecast(len(test_close))

sarimax_rmse = np.sqrt(mean_squared_error(test_close, sarimax_preds))
sarimax_mae = mean_absolute_error(test_close, sarimax_preds)

print("SARIMAX RMSE:", sarimax_rmse)
print("SARIMAX MAE:", sarimax_mae)

# =========================
# 8. LSTM MODEL DEFINITION
# =========================
def build_lstm(units1, units2, dropout):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout),
        LSTM(units2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =========================
# 9. OPTUNA HYPERPARAMETER TUNING
# =========================
def objective(trial):
    units1 = trial.suggest_int("units1", 32, 128, step=32)
    units2 = trial.suggest_int("units2", 16, 64, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = build_lstm(units1, units2, dropout)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        verbose=0
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return loss

print("\nRunning Optuna optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best parameters:", study.best_params)

# =========================
# 10. TRAIN FINAL LSTM MODEL
# =========================
best = study.best_params

final_model = build_lstm(
    best['units1'],
    best['units2'],
    best['dropout']
)

history = final_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# =========================
# 11. EVALUATION
# =========================
lstm_preds = final_model.predict(X_test)

lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
lstm_mae = mean_absolute_error(y_test, lstm_preds)
lstm_mape = np.mean(np.abs((y_test - lstm_preds.flatten()) / y_test)) * 100

print("\nLSTM RMSE:", lstm_rmse)
print("LSTM MAE:", lstm_mae)
print("LSTM MAPE:", lstm_mape)



# 12. INTEGRATED GRADIENTS EXPLAINABILITY
# =========================
print("\nRunning Integrated Gradients analysis...")

@tf.function
def integrated_gradients(model, inputs, baseline, steps=50):
    interpolated_inputs = [
        baseline + (float(i) / steps) * (inputs - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for x in interpolated_inputs:
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x)
        grads.append(tape.gradient(preds, x))

    avg_grads = tf.reduce_mean(tf.stack(grads), axis=0)
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads

baseline = tf.zeros(shape=(1, SEQ_LEN, X.shape[2]))
sample_input = tf.convert_to_tensor(X_test[:1], dtype=tf.float32)

ig_attributions = integrated_gradients(
    final_model,
    sample_input,
    baseline
)

feature_importance = tf.reduce_mean(
    tf.abs(ig_attributions),
    axis=[0, 1]
).numpy()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nIntegrated Gradients Feature Importance:")
print(importance_df)

print("\nPIPELINE COMPLETED SUCCESSFULLY")


features = ['Open', 'High', 'Low', 'Close', 'Volume']
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nSHAP Feature Importance:")
print(importance_df)

print("\nPIPELINE COMPLETED SUCCESSFULLY")
