"""Launcher for local MLflow UI with appropriate environment flags."""

import os
import subprocess
import sys


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "./mlruns",
        "--port",
        port,
    ]
    print("=" * 60)
    print(f"Starting MLflow UI on http://127.0.0.1:{port}")
    print("Backend Store : sqlite:///mlflow.db")
    print("Artifact Root : ./mlruns")
    print("=" * 60)
    subprocess.run(cmd, env=os.environ)


if __name__ == "__main__":
    main()
