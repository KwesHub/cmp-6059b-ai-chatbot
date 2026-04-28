import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_DIR = "data"


def time_to_minutes(time_str: str) -> float:
    """
    Convert a time string like '08:12:00' to minutes past midnight.
    Returns NaN if the string is invalid.
    """
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) >= 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return np.nan


def compute_delay_minutes(planned: str, actual: str) -> float:
    """
    Compute delay in minutes between planned and actual time.
    Positive = late, negative = early.
    Returns NaN if either time is invalid.
    """
    p = time_to_minutes(planned)
    a = time_to_minutes(actual)
    if np.isnan(p) or np.isnan(a):
        return np.nan
    return a - p


def preprocess() -> tuple:
    """
    Load cleaned data, engineer features, and split into
    train and test sets.

    Features (X):
    - planned_arrival_minutes : planned arrival time in mins past midnight
    - planned_departure_minutes : planned departure time in mins
    - day_of_week : 0=Monday, 6=Sunday
    - month : 1-12
    - direction : 0=WAT2WEY, 1=WEY2WAT
    - location_encoded : station CRS code encoded as integer

    Target (y):
    - arrival_delay_minutes : actual - planned arrival (minutes)
    """
    print("Loading cleaned data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))
    print(f"Loaded {len(df):,} rows")

    # ─── Compute delay (target variable y) ───────────────
    print("Computing delay minutes...")
    df["arrival_delay_minutes"] = df.apply(
        lambda row: compute_delay_minutes(
            row["planned_arrival_time"],
            row["actual_arrival_time"]
        ), axis=1
    )

    # Drop rows where we can't compute the delay
    before = len(df)
    df = df.dropna(subset=["arrival_delay_minutes"])
    print(f"Dropped {before - len(df):,} rows with missing actual arrival")

    # ─── Engineer features ────────────────────────────────
    df["planned_arrival_minutes"] = df["planned_arrival_time"].apply(
        time_to_minutes)
    df["planned_departure_minutes"] = df["planned_departure_time"].apply(
        time_to_minutes)

    # Encode direction as binary
    df["direction_encoded"] = (df["direction"] == "WEY2WAT").astype(int)

    # Encode station as integer
    station_codes = sorted(df["location"].unique())
    station_map = {code: i for i, code in enumerate(station_codes)}
    df["location_encoded"] = df["location"].map(station_map)

    # Save the station map so predict.py can use it
    joblib.dump(station_map,
                os.path.join(DATA_DIR, "station_map.pkl"))
    print(f"Station map saved with {len(station_map)} stations")

    # ─── Select features and target ───────────────────────
    FEATURES = [
        "planned_arrival_minutes",
        "planned_departure_minutes",
        "day_of_week",
        "month",
        "direction_encoded",
        "location_encoded"
    ]
    TARGET = "arrival_delay_minutes"

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nFeatures: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\ny statistics:")
    print(f"  Mean delay:   {y.mean():.2f} minutes")
    print(f"  Median delay: {y.median():.2f} minutes")
    print(f"  Max delay:    {y.max():.2f} minutes")
    print(f"  Min delay:    {y.min():.2f} minutes")

    # ─── Split into train and test ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")

    # Save splits
    X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR,  "y_test.csv"),  index=False)
    print("Train/test splits saved to data/")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()