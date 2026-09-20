from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import bmh_apps.benchmark

CONFIG_DIR = Path(bmh_apps.benchmark.__file__).parent / "conf"
EXPERIMENTS = sorted(path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml"))


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name="config", return_hydra_config=True, overrides=["benchmark_path=/benchmark", *overrides])


def test_experiments_are_found():
    assert "mining-f2" in EXPERIMENTS


def test_default_launcher_is_available():
    # Composing fails with "Could not find 'hydra/launcher/joblib'" if the plugin is not installed
    cfg = compose_config("+experiment=mining-f2")

    assert cfg.hydra.launcher["_target_"] == "hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher"
    assert cfg.hydra.launcher.n_jobs == 8


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes_with_objectives_and_material(experiment):
    cfg = compose_config(f"+experiment={experiment}")

    assert len(cfg.optimization.objectives) >= 1
    assert cfg.material_identifier
