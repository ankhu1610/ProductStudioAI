"""FastAPI application package."""

from app.api.app import create_app
from app.api.endpoints import router

__all__ = ["create_app", "router"]
