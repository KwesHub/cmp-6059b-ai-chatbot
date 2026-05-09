import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import math
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras_model import build_keras_model

DATA_DIR  = "data"
MODEL_DIR = "models"


def load_splits():
    """Load the train/test splits from data/"""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def clip_delays(y_train, y_test, max_delay=120):
    """
    Cap delays at ±120 minutes to remove data errors.
    1438-minute delays are midnight wraparounds, not real delays.
    """
    y_train = y_train.clip(-max_delay, max_delay)
    y_test  = y_test.clip(-max_delay, max_delay)
    print(f"Delays clipped to ±{max_delay} minutes")
    return y_train, y_test


def evaluate(name, model, X_test, y_test) -> dict:
    """Evaluate a model and return its metrics."""
    preds = model.predict(X_test)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"{name:25s} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"Model": name, "RMSE": round(rmse, 2),
            "MAE": round(mae, 2), "R2": round(r2, 3)}


def train():
    X_train, X_test, y_train, y_test = load_splits()
    y_train, y_test = clip_delays(y_train, y_test)

    models = {
        "Linear Regression":   LinearRegression(),
        "kNN Regressor (k=5)": KNeighborsRegressor(
            n_neighbors=5, n_jobs=-1),
        "Random Forest":       RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = []
    best_rmse  = float("inf")
    best_model = None
    best_name  = ""

    print("\nTraining models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_test, y_test)
        results.append(metrics)

        # Save every model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved to {path}")

        if metrics["RMSE"] < best_rmse:
            best_rmse  = metrics["RMSE"]
            best_model = model
            best_name  = name

    ## Keras MLP model

    print("\nTraining Keras MLP model...")

    # building the model
    mlp = build_keras_model(X_train.shape[1])

    # training the model using the data
    history = mlp.fit(X_train,y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=1)

    # Asking the model to make a prediction
    mlp_preds = mlp.predict(X_test).flatten()
    # Checking how accurate the prediction is 
    mlp_rmse = math.sqrt(mean_squared_error(y_test, mlp_preds))
    mlp_mae = mean_absolute_error(y_test, mlp_preds)
    mlp_r2 = r2_score(y_test, mlp_preds)

    # Results
    print(f"Keras MLP " f"RMSE={mlp_rmse:.2f}" f"MAE{mlp_mae:.2f}" f"R2={mlp_r2:.3f}")

    #Adds results to the comparision table
    results.append({"Model": "Keras MLP", "RMSE": round(mlp_rmse, 2), "MAE": round(mlp_mae, 2), "R2": round(mlp_r2, 3)})

    # Saves the model
    Keras_path = os.path.join(MODEL_DIR, "keras_mlp.h5")
    mlp.save(Keras_path)

    print(f"Saved keras MLP model to {Keras_path}")

    # Plotting a graph to show models improvement
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train", "validation"])
    plt.title("MLP Loss")
    loss_path = os.path.join(DATA_DIR, "mlp_loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"Saved MLP loss curve to {loss_path}")

    # if Keras MLP is the best model
    if mlp_rmse < best_rmse:
        best_rmse = mlp_rmse
        best_model = mlp
        best_name = "Keras MLP"

    # ─── Save comparison table ────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(DATA_DIR, "model_comparison.csv"), index=False)

    print("\n" + "="*55)
    print("MODEL COMPARISON TABLE")
    print("="*55)
    print(results_df.to_string(index=False))
    print("="*55)
    print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f})")

    # Save best model separately for the chatbot to use
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"Best model saved to {best_path}")

    return results_df


if __name__ == "__main__":
    train()