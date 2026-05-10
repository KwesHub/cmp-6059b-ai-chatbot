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
    Machine learning models need numbers, not strings -- 09:30 becomes 570.
    Returns NaN if the string is invalid so we can drop those rows cleanly.
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
    Compute delay in minutes between planned and actual arrival.
    Positive = late, negative = early (train arrived ahead of schedule).
    Returns NaN if either time is missing.
    """
    p = time_to_minutes(planned)
    a = time_to_minutes(actual)
    if np.isnan(p) or np.isnan(a):
        return np.nan
    return a - p


def preprocess() -> tuple:
    """Load cleaned data, engineer features and split into train/test sets."""
    print("Loading cleaned data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))
    print(f"Loaded {len(df):,} rows")

    # compute the target variable: how many minutes late the train arrived
    print("Computing delay minutes...")
    df["arrival_delay_minutes"] = df.apply(
        lambda row: compute_delay_minutes(
            row["planned_arrival_time"],
            row["actual_arrival_time"]
        ), axis=1
    )

    # drop rows where actual arrival is missing -- we can't train on unknown delays
    before = len(df)
    df = df.dropna(subset=["arrival_delay_minutes"])
    print(f"Dropped {before - len(df):,} rows with missing actual arrival")

    # --- feature engineering ---
    # Convert time strings to minutes-past-midnight so the model sees a continuous number.
    # e.g. 09:30 -> 570. This captures patterns like "trains after 5pm tend to be later".
    df["planned_arrival_minutes"] = df["planned_arrival_time"].apply(time_to_minutes)
    df["planned_departure_minutes"] = df["planned_departure_time"].apply(time_to_minutes)

    # Encode direction as binary: WEY2WAT (Weymouth to Waterloo) = 1, WAT2WEY = 0.
    # The two directions have different delay patterns so this is an important feature.
    df["direction_encoded"] = (df["direction"] == "WEY2WAT").astype(int)

    # Label-encode station CRS codes as integers (0, 1, 2, ...).
    # Models can't process strings, so we map each station to a unique number.
    # The same mapping is saved to disk so predict.py can apply it at inference time.
    station_codes = sorted(df["location"].unique())
    station_map = {code: i for i, code in enumerate(station_codes)}
    df["location_encoded"] = df["location"].map(station_map)

    joblib.dump(station_map, os.path.join(DATA_DIR, "station_map.pkl"))
    print(f"Station map saved with {len(station_map)} stations")

    # six features chosen because they capture the main drivers of train delay:
    # time of day, day of week, month (seasonality), direction and current location
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
    print(f"Target:   {TARGET}")
    print(f"X shape:  {X.shape}")
    print(f"y shape:  {y.shape}")
    print(f"\ny statistics:")
    print(f"  Mean delay:   {y.mean():.2f} minutes")
    print(f"  Median delay: {y.median():.2f} minutes")
    print(f"  Max delay:    {y.max():.2f} minutes")
    print(f"  Min delay:    {y.min():.2f} minutes")

    # 80/20 train-test split with a fixed random seed so results are reproducible.
    # The test set acts as unseen data -- it tells us how well the model generalises.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")

    X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR,  "y_test.csv"),  index=False)
    print("Train/test splits saved to data/")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()
