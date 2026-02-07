import optuna
from src.model import build_lstm

def optimize_lstm(X_train, y_train, X_test, y_test):

    def objective(trial):
        units1 = trial.suggest_int("units1", 32, 128, step=32)
        units2 = trial.suggest_int("units2", 16, 64, step=16)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model = build_lstm(
            (X_train.shape[1], X_train.shape[2]),
            units1,
            units2,
            dropout
        )

        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=15,
            batch_size=32,
            verbose=0
        )

        loss, _ = model.evaluate(X_test, y_test, verbose=0)
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    return study.best_params
