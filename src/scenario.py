from pathlib import Path
import sys, json, joblib, numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data_prep import load_data
from src.features import build_features

ARTIFACTS    = REPO_ROOT / "artifacts"
MODEL_PATH   = ARTIFACTS / "model_ridge.pkl"
METRICS_PATH = ARTIFACTS / "metrics_ridge.json"
TARGET       = "home_price_index"

#  1.0 percentage-point change in the driver maps to this % effect on next month HPI.
W_INTEREST     = -0.8  
W_MORTGAGE     = -0.6   
W_UNEMPLOYMENT = -1.2  

def _recreate_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicate training features: build_features + simple time features + target lags."""
    if "date" in df_raw.columns:
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

    df_fe = build_features(df_raw)

    if "date" in df_fe.columns:
        dt = pd.to_datetime(df_fe["date"])
        df_fe["year"] = dt.dt.year
        df_fe["month"] = dt.dt.month

    df_fe["time_index"] = np.arange(len(df_fe))
    for k in (1, 3, 6, 12):
        df_fe[f"{TARGET}_lag{k}"] = df_fe[TARGET].shift(k)

    return df_fe

def _last_valid_row(df_fe: pd.DataFrame):
    """Return (X_last_row_df, last_actual_float) after dropping rows with any NaNs."""
    dt_cols = list(df_fe.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    X = df_fe.drop(columns=[TARGET] + dt_cols, errors="ignore")
    y = df_fe[TARGET]
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    if X.empty:
        raise RuntimeError("No valid feature rows after preprocessing.")
    return X.iloc[[-1]].copy(), float(y.iloc[-1])

def load_model_and_metrics():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.pipelines.train_model")
    model = joblib.load(MODEL_PATH)
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics

def _get_feature_order(model):
    """Recover training feature order for safe reindexing (Pipeline or plain estimator)."""
    ridge = getattr(model, "named_steps", {}).get("ridge", None)
    if ridge is not None and hasattr(ridge, "feature_names_in_"):
        return list(ridge.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return []

def _crash_probability(pct_change_next: float) -> float:
    """Toy mapping: a -10% 1-month drop -> 100% crash risk; flat/positive -> 0%."""
    return float(min(1.0, max(0.0, -pct_change_next / 0.10)))

def baseline_and_scenario(interest_pp=0.0, mortgage_pp=0.0, unemp_pp=0.0):

    model, _ = load_model_and_metrics()
    raw = load_data(save_processed=False)
    df_fe = _recreate_features(raw)
    X_last, last_actual = _last_valid_row(df_fe)

    # Reindex to training feature order if known
    want = _get_feature_order(model)
    if want:
        X_last = X_last.reindex(columns=want)

    # Baseline from the trained model
    baseline_pred = float(model.predict(X_last)[0])

    #    Total % effect = sum(weight * delta_pp)
    scenario_pct = (
        W_INTEREST     * float(interest_pp)
        + W_MORTGAGE   * float(mortgage_pp)
        + W_UNEMPLOYMENT * float(unemp_pp)
    ) / 100.0  

    scenario_pred = float(baseline_pred * (1.0 + scenario_pct))

    def pct_change(v):
        return (v - last_actual) / last_actual if last_actual else 0.0

    pct_base = pct_change(baseline_pred)
    pct_scn  = pct_change(scenario_pred)

    return {
        "last_actual": last_actual,
        "baseline_pred": baseline_pred,
        "scenario_pred": scenario_pred,
        "pct_change_baseline": pct_base,
        "pct_change_scenario": pct_scn,
        "crash_prob_scenario": _crash_probability(pct_scn),
    }
