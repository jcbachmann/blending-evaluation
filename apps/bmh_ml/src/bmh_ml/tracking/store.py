"""Where the training pipeline keeps everything it persists: datasets, the experiment tracking database with its artifacts, and reports.

The store is outside of the repository on purpose, it grows to gigabytes. It is `~/bmh-ml-store` unless BMH_ML_STORE says otherwise.
"""

import os
from pathlib import Path

STORE_ENVIRONMENT_VARIABLE = "BMH_ML_STORE"
DEFAULT_STORE = "~/bmh-ml-store"


def get_store() -> Path:
    store = Path(os.environ.get(STORE_ENVIRONMENT_VARIABLE) or DEFAULT_STORE).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    return store


def get_subdirectory(name: str) -> Path:
    directory = get_store() / name
    directory.mkdir(exist_ok=True)
    return directory


def get_datasets_directory() -> Path:
    """One directory per dataset, named by its id."""
    return get_subdirectory("datasets")


def get_bundles_directory() -> Path:
    """A bundle names the training, validation and frozen test datasets that belong together."""
    return get_subdirectory("bundles")


def get_reports_directory() -> Path:
    return get_subdirectory("reports")


def get_tracking_uri() -> str:
    """The MLflow tracking database, a SQLite file in the store."""
    return f"sqlite:///{(get_store() / 'mlflow.db').as_posix()}"


def get_artifact_root() -> str:
    """The directory MLflow stores the artifacts of the runs in."""
    root = get_store() / "artifacts"
    root.mkdir(exist_ok=True)
    return root.as_uri()
