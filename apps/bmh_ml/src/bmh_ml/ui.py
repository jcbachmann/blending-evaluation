import argparse
import subprocess
import sys

from bmh_ml.tracking.store import get_artifact_root, get_store, get_tracking_uri


def get_ui_command(host: str = "127.0.0.1", port: int = 5000) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        get_tracking_uri(),
        "--default-artifact-root",
        get_artifact_root(),
        "--host",
        host,
        "--port",
        str(port),
    ]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Starts the MLflow UI for the runs of the store, open http://localhost:<port>")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    print(f"Store {get_store()}, UI at http://{args.host}:{args.port}, stop with Ctrl+C")
    subprocess.run(get_ui_command(args.host, args.port), check=False)  # noqa: S603 - the arguments are the store location and the given host and port


if __name__ == "__main__":
    main()
