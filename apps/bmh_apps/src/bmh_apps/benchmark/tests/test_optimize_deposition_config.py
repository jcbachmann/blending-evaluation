from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import InterpolationResolutionError

import bmh_apps.benchmark

CONFIG_DIR = Path(bmh_apps.benchmark.__file__).parent / "conf"
EXPERIMENTS = sorted(path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml"))


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name="config", return_hydra_config=True, overrides=list(overrides))


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


def test_benchmark_path_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("BMH_BENCHMARK_PATH", "/data/benchmark")

    assert compose_config("+experiment=mining-f2").benchmark_path == "/data/benchmark"


def test_benchmark_path_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BMH_BENCHMARK_PATH", "/data/benchmark")

    assert compose_config("+experiment=mining-f2", "benchmark_path=/other").benchmark_path == "/other"


def test_missing_benchmark_path_is_reported_with_the_name_of_the_environment_variable(monkeypatch):
    monkeypatch.delenv("BMH_BENCHMARK_PATH", raising=False)
    cfg = compose_config("+experiment=mining-f2")

    with pytest.raises(InterpolationResolutionError, match="BMH_BENCHMARK_PATH"):
        _ = cfg.benchmark_path


def test_benchmark_path_is_not_part_of_the_run_directory_name(monkeypatch):
    monkeypatch.setenv("BMH_BENCHMARK_PATH", "/data/benchmark")
    cfg = compose_config("+experiment=mining-f2", "benchmark_path=/data/other", "+run=3", "optimization.max_evaluations=100")

    assert cfg.hydra.job.config.override_dirname.exclude_keys == ["benchmark_path"]
