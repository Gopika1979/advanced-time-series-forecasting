import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

def sarimax_baseline(train, test):
    model = SARIMAX(train, order=(5,1,0))
    results = model.fit(disp=False)
    preds = results.forecast(len(test))

    rmse = np.sqrt(mean_squared_error(test, preds))
    mae = mean_absolute_error(test, preds)

    return rmse, mae
