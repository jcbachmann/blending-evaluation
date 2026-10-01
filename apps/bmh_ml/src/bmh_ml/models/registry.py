import importlib
import json
from pathlib import Path

from bmh_ml.models.base import MODEL_FILE, Model

# The classes are imported when they are used, so a missing optional package only affects the models that need it
MODELS = {
    "mean": "bmh_ml.models.baselines:MeanModel",
    "ridge": "bmh_ml.models.baselines:RidgeModel",
    "lightgbm": "bmh_ml.models.baselines:LightGBMModel",
    "mlp": "bmh_ml.models.keras_models:MLPModel",
    "legacy_lstm": "bmh_ml.models.keras_models:LegacyLSTMModel",
    "profile_mlp": "bmh_ml.models.keras_models:ProfileMLPModel",
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', use one of {sorted(MODELS)}")
    module_name, _, class_name = MODELS[name].partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def create_model(name: str, **params) -> Model:
    return get_model_class(name)(**params)


def load_model(directory: Path) -> Model:
    stored = json.loads((directory / MODEL_FILE).read_text())
    model = create_model(stored["name"], **stored["params"])
    model.load_files(directory)
    return model
