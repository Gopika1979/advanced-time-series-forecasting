from src.data_preprocessing import load_data, preprocess
from src.baseline import sarimax_baseline
from src.train import optimize_lstm
from src.model import build_lstm
from src.explainability import integrated_gradients

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

SEQ_LEN = 60

df = load_data()
X, y, _ = preprocess(df, SEQ_LEN)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

rmse_sarimax, mae_sarimax = sarimax_baseline(
    df['Close'][:split+SEQ_LEN],
    df['Close'][split+SEQ_LEN:]
)

best = optimize_lstm(X_train, y_train, X_test, y_test)

model = build_lstm(
    (X_train.shape[1], X_train.shape[2]),
    best['units1'],
    best['units2'],
    best['dropout']
)

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("\nLSTM RMSE:", rmse)
print("LSTM MAE:", mae)

importance_df = integrated_gradients(model, X_test, SEQ_LEN)
print("\nFeature Importance:\n", importance_df)
