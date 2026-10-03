from bmh_ml.tracking import store


def test_the_store_is_taken_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "my-store"))

    assert store.get_store() == tmp_path / "my-store"
    assert (tmp_path / "my-store").is_dir()  # created on first use


def test_the_default_store_is_in_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("BMH_ML_STORE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows

    assert store.get_store() == tmp_path / "bmh-ml-store"


def test_an_empty_variable_means_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert store.get_store() == tmp_path / "bmh-ml-store"


def test_subdirectories_are_created_inside_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path))

    for get_directory, name in [(store.get_datasets_directory, "datasets"), (store.get_bundles_directory, "bundles"), (store.get_reports_directory, "reports")]:
        assert get_directory() == tmp_path / name
        assert (tmp_path / name).is_dir()


def test_tracking_database_is_a_sqlite_file_in_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path))
    monkeypatch.delenv("BMH_ML_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////elsewhere/mlflow.db")  # set by MLflow for an earlier store, must not count

    assert store.get_tracking_uri() == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_a_tracking_server_replaces_the_database_of_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path))
    monkeypatch.setenv("BMH_ML_TRACKING_URI", "http://127.0.0.1:5055")

    assert store.get_tracking_uri() == "http://127.0.0.1:5055"


def test_artifacts_are_stored_in_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path))

    assert store.get_artifact_root() == (tmp_path / "artifacts").as_uri()
    assert (tmp_path / "artifacts").is_dir()
