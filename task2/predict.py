import joblib
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")
 

MODEL_DIR = "models"
DATA_DIR  = "data"

# ─── Load model and station map once at import time ──────
_model       = None
_station_map = None


def _load():
    global _model, _station_map
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        _model = joblib.load(model_path)
    if _station_map is None:
        map_path = os.path.join(DATA_DIR, "station_map.pkl")
        _station_map = joblib.load(map_path)


def time_to_minutes(time_str: str) -> float:
    """Convert HH:MM or HH:MM:SS to minutes past midnight."""
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) >= 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return 0.0


def predict_delay(
    current_station_crs: str,
    destination_crs: str,
    planned_arrival_time: str,
    planned_departure_time: str,
    direction: str,
    day_of_week: int,
    month: int
) -> dict:
    """
    Predict the delay in minutes at the destination station.

    Args:
        current_station_crs   : 3-letter CRS code of current station
        destination_crs       : 3-letter CRS code of destination
        planned_arrival_time  : planned arrival at destination "HH:MM"
        planned_departure_time: planned departure from current station
        direction             : "WEY2WAT" or "WAT2WEY"
        day_of_week           : 0=Monday, 6=Sunday
        month                 : 1-12

    Returns dict with:
        predicted_delay_minutes : float
        predicted_arrival       : string "HH:MM"
        confidence              : "low" / "medium" / "high"
    """
    _load()

    # Encode direction
    direction_encoded = 1 if direction == "WEY2WAT" else 0

    # Encode station
    location_encoded = _station_map.get(current_station_crs, 0)

    # Convert times to minutes
    arr_mins  = time_to_minutes(planned_arrival_time)
    dep_mins  = time_to_minutes(planned_departure_time)

    # Build feature row
    X = pd.DataFrame([{
        "planned_arrival_minutes":   arr_mins,
        "planned_departure_minutes": dep_mins,
        "day_of_week":               day_of_week,
        "month":                     month,
        "direction_encoded":         direction_encoded,
        "location_encoded":          location_encoded
    }])

    # Predict
    delay = float(_model.predict(X)[0])
    delay = round(delay, 1)

    # Compute predicted arrival time
    predicted_arrival_mins = arr_mins + delay
    predicted_hour = int(predicted_arrival_mins // 60) % 24
    predicted_min  = int(predicted_arrival_mins % 60)
    predicted_arrival = f"{predicted_hour:02d}:{predicted_min:02d}"

    # Confidence based on delay magnitude
    abs_delay = abs(delay)
    if abs_delay < 3:
        confidence = "high"
    elif abs_delay < 10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "predicted_delay_minutes": delay,
        "predicted_arrival":       predicted_arrival,
        "confidence":              confidence
    }


# ─── Manual test ─────────────────────────────────────────
if __name__ == "__main__":
    print("Test: Train delayed at Southampton, heading to London Waterloo")
    print("Direction: WEY2WAT, Monday, July")
    print()

    result = predict_delay(
        current_station_crs="SOU",
        destination_crs="WAT",
        planned_arrival_time="11:30",
        planned_departure_time="09:00",
        direction="WEY2WAT",
        day_of_week=0,
        month=7
    )

    print(f"Predicted delay:   {result['predicted_delay_minutes']} minutes")
    print(f"Predicted arrival: {result['predicted_arrival']}")
    print(f"Confidence:        {result['confidence']}")