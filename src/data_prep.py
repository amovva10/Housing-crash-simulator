import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "US_House_Price.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "US_House_Price_clean.csv"

# Load raw CSV, parse dates, rename some columns, save new data to processed/
def load_data(path: Path = RAW_PATH, save_processed: bool = True) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, parse_dates=["DATE"])
    if "DATE" not in df.columns:
        raise KeyError("Expected 'DATE' column in CSV.")
    df.set_index("DATE", inplace=True)

    # Rename columns for clarity
    df.rename(
        columns={
            "total_houses": "households",
            "house_for_sale_or_sold": "housing_inventory",
            "const_price_index": "construction_price_index",
            "total_const_spending": "construction_spending_change",
        },
        inplace=True,
    )

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected:\n", missing[missing > 0])
    else:
        print("No missing values found.")

    # Save processed file
    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH)
        print(f"Processed file saved to: {PROCESSED_PATH}")

    return df

if __name__ == "__main__":
    df = load_data(save_processed=True)
    print(df.head())
