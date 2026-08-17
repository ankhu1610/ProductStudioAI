"""Launcher for local MLflow UI with appropriate environment flags."""

import os
import subprocess
import sys


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    port = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("sqlite") else "5000"
    
    # Check if outputs/mlflow.db exists (Docker output path), else fallback to mlflow.db
    backend_db = "sqlite:///outputs/mlflow.db" if os.path.exists("outputs/mlflow.db") else "sqlite:///mlflow.db"
    
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_db,
        "--default-artifact-root",
        "./mlruns",
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--workers",
        "1",
    ]
    print("=" * 60)
    print(f"Starting MLflow UI on http://127.0.0.1:{port}")
    print(f"Backend Store : {backend_db}")
    print("Artifact Root : ./mlruns")
    print("=" * 60)
    subprocess.run(cmd, env=os.environ)


if __name__ == "__main__":
    main()
