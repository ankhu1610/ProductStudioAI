"""FastAPI application factory and Gradio UI mounting."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.api.endpoints import router as api_router
from app.core.config import Settings, get_settings
from app.services.database import DatabaseService
from app.services.pipeline import ProductStudioPipeline
from app.services.tracking import MLflowTracker
from app.ui.interface import build_gradio_app


def create_app(
    settings: Settings | None = None,
    pipeline: ProductStudioPipeline | None = None,
    db: DatabaseService | None = None,
    tracker: MLflowTracker | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="ProductStudio AI API",
        description="Mask-Aware Latent Diffusion Platform for E-Commerce Ad Creative Generation",
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # State dependencies
    app.state.settings = settings
    app.state.pipeline = pipeline or ProductStudioPipeline(settings=settings)
    app.state.db_service = db or DatabaseService(settings=settings)
    app.state.tracker = tracker or MLflowTracker(settings=settings)

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API routes
    app.include_router(api_router)

    # Root redirect to UI
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/ui")

    # Mount Gradio UI at /ui
    gradio_demo = build_gradio_app(settings=settings, pipeline=app.state.pipeline, db=app.state.db_service)
    app = gr.mount_gradio_app(app, gradio_demo, path="/ui")

    return app
