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

from bmh_ml.settings import DATA_DIR, MATERIAL_LENGTH, MODEL_F1_NAME, MODEL_F2_NAME, TRAINING_DATA_FILE, add_model_set_argument, get_model_file, get_scaler_file
from bmh_ml.training_data import load_training_data


def preprocess_data(data: pd.DataFrame):
    x = data.drop(columns=["f1", "f2"]).to_numpy()
    y = data[["f1", "f2"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x.reshape((x.shape[0], 1, x.shape[1])), y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def train_model(x_train, y_train, x_test, y_test, model_name: str, model_set: str, epochs: int):
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

    model.save(get_model_file(model_name, model_set))


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info(f"Training model set '{args.model_set}' with {args.training_data}")
    data = load_training_data(args.training_data)
    x_train, x_test, y_train, y_test, scaler = preprocess_data(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(get_scaler_file(args.model_set), "wb") as f:
        pickle.dump(scaler, f)

    train_model(x_train, y_train[:, 0], x_test, y_test[:, 0], MODEL_F1_NAME, args.model_set, args.epochs)
    # F2 (reclaim volume deviation) only depends on the deposition, the material volume per time is constant
    train_model(x_train[:, :, MATERIAL_LENGTH:], y_train[:, 1], x_test[:, :, MATERIAL_LENGTH:], y_test[:, 1], MODEL_F2_NAME, args.model_set, args.epochs)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs of each model")
    parser.add_argument("--training-data", default=TRAINING_DATA_FILE, help="CSV file with the training data")
    # The name is required, so a model set is never trained by accident and models trained on one dataset are not stored under the name of another one
    add_model_set_argument(parser, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
