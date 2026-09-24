import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH
from bmh_ml.tracking.environment import get_code_version

SCOPE_FIXED_MATERIAL = "S1"
SCOPE_GENERAL_MATERIAL = "S2"
SCOPES = (SCOPE_FIXED_MATERIAL, SCOPE_GENERAL_MATERIAL)


@dataclass
class Dataset:
    """Inputs and labels of one dataset.

    `material` has one row per sample, or a single row when all samples share the material. `y` holds F1 and F2, the mean of `repeats`
    simulations of the same input; `y_noise_sd` is the standard deviation of those simulations (only if repeated).
    """

    name: str
    generator: str
    seed: int
    repeats: int
    material: np.ndarray
    deposition: np.ndarray
    y: np.ndarray
    y_noise_sd: np.ndarray | None = None
    settings: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    code_version: str = field(default_factory=get_code_version)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.material.ndim != 2 or self.material.shape[1] != MATERIAL_LENGTH:
            raise ValueError(f"material must have shape (rows, {MATERIAL_LENGTH}), got {self.material.shape}")
        if self.deposition.ndim != 2 or self.deposition.shape[1] != DEPOSITION_LENGTH:
            raise ValueError(f"deposition must have shape (rows, {DEPOSITION_LENGTH}), got {self.deposition.shape}")
        if self.material.shape[0] not in (1, len(self.deposition)):
            raise ValueError("material must have one row or one row per sample")
        if self.y.shape != (len(self.deposition), 2):
            raise ValueError(f"y must have shape ({len(self.deposition)}, 2), got {self.y.shape}")

    def __len__(self) -> int:
        return len(self.deposition)

    def full_material(self) -> np.ndarray:
        """The material of every sample, shape (rows, MATERIAL_LENGTH)."""
        return np.broadcast_to(self.material, (len(self), MATERIAL_LENGTH))

    def features(self, scope: str) -> np.ndarray:
        """Model input: the deposition for S1, the material followed by the deposition for S2."""
        return make_features(scope, self.material, self.deposition)

    def content_hash(self) -> str:
        """Hash of the numbers, so that the same content always has the same id."""
        digest = hashlib.sha256()
        for name, array in (("material", self.material), ("deposition", self.deposition), ("y", self.y), ("y_noise_sd", self.y_noise_sd)):
            if array is not None:
                digest.update(f"{name}{array.dtype}{array.shape}".encode())
                digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def dataset_id(self) -> str:
        return f"{self.name}-{self.content_hash()[:10]}"

    def manifest(self) -> dict:
        return {
            "id": self.dataset_id(),
            "name": self.name,
            "generator": self.generator,
            "size": len(self),
            "seed": self.seed,
            "repeats": self.repeats,
            "settings": self.settings,
            "source": self.source,
            "code_version": self.code_version,
            "created": self.created,
            "content_hash": self.content_hash(),
            "material_rows": int(self.material.shape[0]),
        }


def make_features(scope: str, material: np.ndarray, deposition: np.ndarray) -> np.ndarray:
    """Model input: the deposition for S1, the material followed by the deposition for S2. `material` is one row for all or one row each."""
    if scope == SCOPE_FIXED_MATERIAL:
        return deposition
    if scope == SCOPE_GENERAL_MATERIAL:
        return np.hstack([np.broadcast_to(material, (len(deposition), MATERIAL_LENGTH)), deposition])
    raise ValueError(f"Unknown scope '{scope}', use one of {SCOPES}")


def concatenate_datasets(name: str, datasets: list[Dataset]) -> Dataset:
    """The rows of several datasets, in order. The material stays a single row if all datasets share it.

    The labels of the datasets may be the mean of different numbers of simulations, the result has the repeats of the first dataset.
    """
    if len(datasets) == 1:
        return datasets[0]
    materials = [dataset.material for dataset in datasets]
    if all(material.shape[0] == 1 for material in materials) and all(np.array_equal(material, materials[0]) for material in materials):
        material = materials[0]
    else:
        material = np.vstack([dataset.full_material() for dataset in datasets])
    first = datasets[0]
    return Dataset(
        name=name,
        generator="+".join(dict.fromkeys(dataset.generator for dataset in datasets)),
        seed=first.seed,
        repeats=first.repeats,
        material=material,
        deposition=np.vstack([dataset.deposition for dataset in datasets]),
        y=np.vstack([dataset.y for dataset in datasets]),
        settings=first.settings,
        source={"datasets": [dataset.dataset_id() for dataset in datasets]},
    )


def manifest_to_json(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True)
