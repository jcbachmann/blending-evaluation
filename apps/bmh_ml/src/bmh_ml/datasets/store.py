import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from bmh_ml.datasets.manifest import Dataset, manifest_to_json
from bmh_ml.tracking.store import get_bundles_directory, get_datasets_directory


def save_dataset(dataset: Dataset) -> str:
    """Stores the dataset under its id, identical content is stored once. Returns the id."""
    dataset_id = dataset.dataset_id()
    directory = get_datasets_directory() / dataset_id
    if directory.exists():
        return dataset_id

    temporary = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    arrays = {"material": dataset.material, "deposition": dataset.deposition, "y": dataset.y}
    if dataset.y_noise_sd is not None:
        arrays["y_noise_sd"] = dataset.y_noise_sd
    np.savez_compressed(temporary / "data.npz", **arrays)
    (temporary / "manifest.json").write_text(manifest_to_json(dataset.manifest()))
    temporary.rename(directory)
    return dataset_id


def load_dataset(dataset_id: str) -> Dataset:
    directory = get_datasets_directory() / dataset_id
    if not directory.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found in {directory.parent}")
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "data.npz") as data:
        arrays = {name: data[name] for name in data.files}
    dataset = Dataset(
        name=manifest["name"],
        generator=manifest["generator"],
        seed=manifest["seed"],
        repeats=manifest["repeats"],
        material=arrays["material"],
        deposition=arrays["deposition"],
        y=arrays["y"],
        y_noise_sd=arrays.get("y_noise_sd"),
        settings=manifest["settings"],
        source=manifest["source"],
        code_version=manifest["code_version"],
        created=manifest["created"],
    )
    if dataset.content_hash() != manifest["content_hash"]:
        raise ValueError(f"Dataset '{dataset_id}' is damaged, its content does not match its hash")
    return dataset


def list_datasets() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(get_datasets_directory().glob("*/manifest.json"))]


@dataclass
class Bundle:
    """The datasets that belong together: training, validation and the frozen test sets, and the scope they are meant for."""

    name: str
    scope: str
    train: str
    val: str
    tests: dict[str, str]
    notes: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def get_bundle_file(name: str):
    return get_bundles_directory() / f"{name}.json"


def bundle_exists(name: str) -> bool:
    return get_bundle_file(name).exists()


def save_bundle(bundle: Bundle) -> None:
    """Bundles are frozen: an existing name is never overwritten, so results that refer to it stay comparable."""
    path = get_bundle_file(bundle.name)
    if path.exists():
        raise FileExistsError(f"Bundle '{bundle.name}' already exists and is frozen, choose another name")
    path.write_text(json.dumps(bundle.__dict__, indent=2, sort_keys=True))


def load_bundle(name: str) -> Bundle:
    path = get_bundle_file(name)
    if not path.exists():
        raise FileNotFoundError(f"Bundle '{name}' not found in {path.parent}")
    return Bundle(**json.loads(path.read_text()))


def list_bundles() -> list[str]:
    return sorted(path.stem for path in get_bundles_directory().glob("*.json"))
