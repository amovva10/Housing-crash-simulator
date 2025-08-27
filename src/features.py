import pandas as pd
from typing import Iterable, Tuple, Optional, Sequence

DEFAULT_LAG_COLS: Sequence[str] = [
    "mortgage_rate",
    "interest_rate",
    "unemployment_rate",
    "housing_inventory",
    "GDP",
]

def add_lags(
    df: pd.DataFrame,
    cols: Iterable[str] = DEFAULT_LAG_COLS,
    lags: Tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            for L in lags:
                out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def build_features(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    lags: Tuple[int, ...] = (1, 3),
    dropna: bool = True,
    sort_index: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if sort_index and not out.index.is_monotonic_increasing:
        out = out.sort_index()

    use_cols = list(cols) if cols is not None else [c for c in DEFAULT_LAG_COLS if c in out.columns]
    out = add_lags(out, cols=use_cols, lags=lags)

    if dropna:
        new_cols = [f"{c}_lag{L}" for c in use_cols for L in lags if f"{c}_lag{L}" in out.columns]
        out = out.dropna(subset=new_cols)

    return out

if __name__ == "__main__":
    from src.data_prep import load_data
    df = load_data(save_processed=False)
    df_fe = build_features(df)
    print("Lag columns preview:\n", df_fe.filter(regex=r"_lag(1|3)$").head())
