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
    print(f"Delays clipped to +/-{max_delay} minutes")
    return y_train, y_test


def build_mlp(input_dim: int):
    """
    Build a Multi-Layer Perceptron (MLP) for delay regression.

    Architecture: input -> Dense(128) -> Dense(64) -> Dense(32) -> output(1)

    ReLU activation introduces non-linearity so the network can learn
    patterns that Linear Regression cannot (e.g. delays only getting bad
    past a certain time of day).

    Dropout layers randomly switch off 10% of neurons during each training
    step. This is a regularisation technique that prevents overfitting --
    the network can't memorise the training data if different neurons are
    disabled each time.

    The output layer has no activation because this is a regression task
    (predicting a continuous value, not a class label).
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
        layers.Dense(1)          # single output: predicted delay in minutes
    ], name="delay_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",              # mean squared error penalises large errors heavily
        metrics=["mae"]          # we also track MAE as it's easier to interpret
    )
    return model


def train():
    X_train, X_test, y_train, y_test = load_splits()
    y_train, y_test = clip_delays(y_train, y_test)

    # StandardScaler transforms each feature to have mean=0 and std=1.
    # Neural networks use gradient descent which is sensitive to feature magnitude --
    # without scaling, a feature in the thousands (e.g. minutes past midnight)
    # would dominate the gradient updates and slow down training.
    # The scaler is fit on training data only, then applied to the test set,
    # to avoid leaking any test information into training.
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # save the scaler so predict.py can apply the same transformation at inference time
    scaler_path = os.path.join(MODEL_DIR, "keras_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    from tensorflow import keras
    model = build_mlp(input_dim=X_train_s.shape[1])
    model.summary()

    print("\nTraining Keras MLP...")

    # EarlyStopping monitors validation loss and stops training when it stops
    # improving, then restores the weights from the best epoch.
    # This prevents overfitting without having to guess the number of epochs.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train_s, y_train,
        validation_split=0.1,   # 10% of training data used to monitor overfitting
        epochs=30,
        batch_size=1024,
        callbacks=[early_stop],
        verbose=1
    )

    preds = model.predict(X_test_s, verbose=0).flatten()
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print("\n" + "=" * 55)
    print("KERAS MLP RESULTS")
    print("=" * 55)
    print(f"  RMSE : {rmse:.2f} minutes")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  R2   : {r2:.3f}")
    print("=" * 55)

    model_path = os.path.join(MODEL_DIR, "keras_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # append results to model_comparison.csv so we can compare Keras against
    # the sklearn models trained in train_models.py
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
