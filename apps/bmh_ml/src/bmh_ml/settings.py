"""Settings shared by all scripts, they must match the parameters the training data was generated with (see the defaults of generate_training_data)."""

MATERIAL_LENGTH: int = 50  # Length of material variables array
DEPOSITION_LENGTH: int = 20  # Length of deposition variables array
BED_SIZE_X: int = 59  # Bed size in X dimension
BED_SIZE_Z: int = 20  # Bed size in Z dimension
TOTAL_VOLUME: int = 2500  # Total volume for simulation
MATERIAL_MIN: int = 5
MATERIAL_MAX: int = 10

# Deposition positions are limited to the core of the stockpile
X_MIN: float = 0.5 * BED_SIZE_Z
X_MAX: float = BED_SIZE_X - X_MIN

# All scripts read and write relative to the working directory
DATA_DIR = "data"
OUTPUT_DIR = "output"
TRAINING_DATA_FILE = f"{DATA_DIR}/training_data.csv"
SCALER_FILE = f"{DATA_DIR}/scaler.pkl"
MODEL_F1_NAME = "lstm_model_f1_random_training_data"
MODEL_F2_NAME = "lstm_model_f2_random_training_data"


def get_model_file(model_name: str) -> str:
    return f"{DATA_DIR}/{model_name}.keras"
