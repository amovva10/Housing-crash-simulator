from pathlib import Path
import json, joblib, pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_prep import load_data
from src.features import build_features

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = load_data(save_processed=False)

    # Raw is in chronological order
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    df_fe = build_features(df)

    # Set target 
    TARGET = "home_price_index"
    y = df_fe[TARGET]

    # Add safe time features and target lags
    if "date" in df_fe.columns:
        dt = pd.to_datetime(df_fe["date"])
        df_fe["year"] = dt.dt.year
        df_fe["month"] = dt.dt.month
    df_fe["time_index"] = np.arange(len(df_fe))
    for k in (1, 3, 6, 12):
        df_fe[f"{TARGET}_lag{k}"] = df_fe[TARGET].shift(k)

    datetime_cols = list(df_fe.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    X = df_fe.drop(columns=[TARGET] + datetime_cols, errors="ignore")

    # Take out NaN values
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=40
    )

    # Train Ridge Regression with time series cross-validation 
    alphas = np.logspace(-3, 2, 40)  # 0.001 → 100
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5)))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    print("Ridge Regression: MAE:", round(mae, 3), "R²:", round(r2, 3))
    print("Train R²:", round(model.score(X_train, y_train), 3),
          "Test R²:", round(model.score(X_test, y_test), 3))
    print("Selected alpha:", float(model.named_steps["ridge"].alpha_))

    # Save model and metrics
    joblib.dump(model, ARTIFACTS / "model_ridge.pkl")
    with open(ARTIFACTS / "metrics_ridge.json", "w") as f:
        json.dump({"MAE": float(mae), "R2": float(r2), "alpha": float(model.named_steps["ridge"].alpha_)}, f, indent=2)

    print("Saved model → artifacts/model_ridge.pkl")
    print("Saved metrics → artifacts/metrics_ridge.json")
