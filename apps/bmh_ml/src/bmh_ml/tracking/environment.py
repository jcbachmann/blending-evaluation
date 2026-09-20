"""What identifies the code and the machine a result was produced with."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def get_repository_root() -> Path | None:
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def run_git(git: str, root: Path, *arguments: str) -> str:
    # Only fixed arguments are passed, the executable is resolved with shutil.which
    return subprocess.run([git, *arguments], cwd=root, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def get_code_version() -> str:
    """The git commit, with `+dirty` if there are uncommitted changes, or `unknown` outside of a git repository."""
    root = get_repository_root()
    git = shutil.which("git")
    if root is None or git is None:
        return "unknown"
    try:
        commit = run_git(git, root, "rev-parse", "--short", "HEAD")
        dirty = run_git(git, root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+dirty" if dirty else commit


def get_lock_hash() -> str:
    """A short hash of uv.lock, which pins all package versions."""
    root = get_repository_root()
    lock_file = None if root is None else root / "uv.lock"
    if lock_file is None or not lock_file.exists():
        return "unknown"
    return hashlib.sha256(lock_file.read_bytes()).hexdigest()[:12]


def get_hardware() -> dict[str, int]:
    return {"cpu_count": os.cpu_count() or 0}
