from bmh_ml.tracking import environment


def test_the_code_version_is_a_commit_or_unknown():
    version = environment.get_code_version()

    assert version == "unknown" or version.removesuffix("+dirty").isalnum()


def test_the_code_version_is_unknown_without_git(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _name: None)

    assert environment.get_code_version() == "unknown"


def test_the_code_version_is_unknown_outside_of_a_repository(monkeypatch):
    monkeypatch.setattr(environment, "get_repository_root", lambda: None)

    assert environment.get_code_version() == "unknown"


def test_the_lock_hash_is_short_or_unknown():
    lock_hash = environment.get_lock_hash()

    assert lock_hash == "unknown" or len(lock_hash) == 12


def test_the_hardware_has_a_cpu_count():
    assert environment.get_hardware()["cpu_count"] >= 1
