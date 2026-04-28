import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ─── File paths ──────────────────────────────────────────
DATA_DIR = "data"

WEY2WAT_FILES = [
    "2022_WEY2WAT.xlsx",
    "2023_WEY2WAT.xlsx",
    "2024_WEY2WAT.xlsx",
    "2025_WEY2WAT.xlsx",
]

WAT2WEY_FILES = [
    "2022_WAT2WEY.xlsx",
    "2023_WAT2WEY.xlsx",
    "2024_WAT2WEY.xlsx",
    "2025_WAT2WEY.xlsx",
]


# ─── Load and tag all files ──────────────────────────────
def load_all_files() -> pd.DataFrame:
    """Load all 8 xlsx files and combine into one DataFrame."""
    frames = []

    for filename in WEY2WAT_FILES:
        path = os.path.join(DATA_DIR, filename)
        print(f"Loading {filename}...")
        df = pd.read_excel(path)
        df["direction"] = "WEY2WAT"
        frames.append(df)

    for filename in WAT2WEY_FILES:
        path = os.path.join(DATA_DIR, filename)
        print(f"Loading {filename}...")
        df = pd.read_excel(path)
        df["direction"] = "WAT2WEY"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined):,}")
    return combined


# ─── Clean the data ──────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the raw data."""

    print("\nCleaning data...")

    # Drop rows where planned times are missing
    before = len(df)
    df = df.dropna(subset=["planned_arrival_time", "planned_departure_time"])
    print(f"Dropped {before - len(df):,} rows with missing planned times")

    # Convert time columns to strings so we can parse them
    time_cols = [
        "planned_arrival_time", "planned_departure_time",
        "actual_arrival_time",  "actual_departure_time"
    ]
    for col in time_cols:
        df[col] = df[col].astype(str)

    # Convert date_of_service to datetime
    df["date_of_service"] = pd.to_datetime(
        df["date_of_service"], errors="coerce"
    )

    # Drop rows where date conversion failed
    before = len(df)
    df = df.dropna(subset=["date_of_service"])
    print(f"Dropped {before - len(df):,} rows with invalid dates")

    # Extract day of week and month
    df["day_of_week"] = df["date_of_service"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"]       = df["date_of_service"].dt.month

    print(f"\nFinal row count: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    return df


# ─── Save cleaned data ───────────────────────────────────
def save(df: pd.DataFrame):
    output_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all_files()
    df = clean(df)
    save(df)
    print("\nDone. Sample:")
    print(df[["rid", "date_of_service", "location", "direction",
              "planned_arrival_time", "actual_arrival_time",
              "day_of_week", "month"]].head(5))