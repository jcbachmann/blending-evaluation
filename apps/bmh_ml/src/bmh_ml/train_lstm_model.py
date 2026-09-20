import argparse
import logging
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from bmh_ml.settings import DATA_DIR, MATERIAL_LENGTH, MODEL_F1_NAME, MODEL_F2_NAME, SCALER_FILE, get_model_file
from bmh_ml.training_data import load_training_data


def preprocess_data(data: pd.DataFrame):
    x = data.drop(columns=["f1", "f2"]).to_numpy()
    y = data[["f1", "f2"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x.reshape((x.shape[0], 1, x.shape[1])), y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def train_model(x_train, y_train, x_test, y_test, model_name: str, epochs: int):
    model = Sequential(
        [
            Input(shape=(x_train.shape[1], x_train.shape[2])),
            LSTM(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1,
    )
    logging.info(f"Final training loss for {model_name}: {history.history['loss'][-1]:.4f}")
    logging.info(f"Final validation loss for {model_name}: {history.history['val_loss'][-1]:.4f}")

    test_loss = model.evaluate(x_test, y_test)
    logging.info(f"Test MSE for {model_name}: {test_loss}")

    model.save(get_model_file(model_name))


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    data = load_training_data()
    x_train, x_test, y_train, y_test, scaler = preprocess_data(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    train_model(x_train, y_train[:, 0], x_test, y_test[:, 0], MODEL_F1_NAME, args.epochs)
    # F2 (reclaim volume deviation) only depends on the deposition, the material volume per time is constant
    train_model(x_train[:, :, MATERIAL_LENGTH:], y_train[:, 1], x_test[:, :, MATERIAL_LENGTH:], y_test[:, 1], MODEL_F2_NAME, args.epochs)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs of each model")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
