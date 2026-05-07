import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR  = "data"
MODEL_DIR = "models"


def load_splits():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def clip_delays(y_train, y_test, max_delay=120):
    y_train = y_train.clip(-max_delay, max_delay)
    y_test  = y_test.clip(-max_delay, max_delay)
    print(f"Delays clipped to ±{max_delay} minutes")
    return y_train, y_test


def build_mlp(input_dim: int):
    """
    Multi-Layer Perceptron for delay regression.
    Architecture: input(6) -> 128 -> 64 -> 32 -> 1
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ], name="delay_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train():
    # ── Load and clip ─────────────────────────────────────
    X_train, X_test, y_train, y_test = load_splits()
    y_train, y_test = clip_delays(y_train, y_test)

    # ── Scale features ────────────────────────────────────
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Save scaler so predict.py can use it at inference time
    scaler_path = os.path.join(MODEL_DIR, "keras_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # ── Build and summarise ───────────────────────────────
    from tensorflow import keras
    model = build_mlp(input_dim=X_train_s.shape[1])
    model.summary()

    # ── Train ─────────────────────────────────────────────
    print("\nTraining Keras MLP...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train_s, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=1024,
        callbacks=[early_stop],
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────
    preds = model.predict(X_test_s, verbose=0).flatten()
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print("\n" + "=" * 55)
    print("KERAS MLP RESULTS")
    print("=" * 55)
    print(f"  RMSE : {rmse:.2f} minutes")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  R²   : {r2:.3f}")
    print("=" * 55)

    # ── Save model ────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "keras_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # ── Append to model_comparison.csv ───────────────────
    comparison_path = os.path.join(DATA_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)
        df = df[df["Model"] != "Keras MLP"]
    else:
        df = pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])

    new_row = pd.DataFrame([{
        "Model": "Keras MLP",
        "RMSE": round(rmse, 2),
        "MAE":  round(mae,  2),
        "R2":   round(r2,   3)
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(comparison_path, index=False)
    print(f"\nFull model comparison:")
    print(df.to_string(index=False))

    return model


if __name__ == "__main__":
    train()
