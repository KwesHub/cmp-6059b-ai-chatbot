from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   

def build_keras_model(input_dim):
    # Build a simple MLP model with Keras
    model = Sequential()

    # First hidden Layer
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    # Second hidden Layer
    model.add(Dense(32, activation="relu"))
    # Output Layer
    model.add(Dense(1))

    # Compiiling the model and preparing it for training
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model