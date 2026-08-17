"""Main entrypoint to launch ProductStudio AI server (FastAPI + Gradio UI)."""

from __future__ import annotations

import argparse
import uvicorn

from app.api.app import create_app
from app.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the ProductStudio AI server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()
    settings = get_settings()

    print("=" * 65)
    print("ProductStudio AI Server Starting...")
    print(f"Device        : {settings.device} (Precision: {settings.precision})")
    print(f"Model ID      : {settings.model_id}")
    print(f"REST API Docs : http://{args.host}:{args.port}/docs")
    print(f"Gradio Web UI : http://{args.host}:{args.port}/ui")
    print("=" * 65)

    app = create_app(settings=settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
