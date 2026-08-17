from unittest.mock import MagicMock
from PIL import Image
import pytest

from app.core.config import Settings
from app.services.database import DatabaseService
from app.ui.interface import PRESETS, build_gradio_app


def test_gradio_app_structure(tmp_path):
    settings = Settings(device="cpu", output_dir=tmp_path)
    db = DatabaseService(db_path=tmp_path / "test.db")
    mock_pipeline = MagicMock()

    demo = build_gradio_app(settings=settings, pipeline=mock_pipeline, db=db)
    assert demo.title == "ProductStudio AI - E-Commerce Ad Studio"
    assert len(PRESETS) >= 8


def test_presets_exist():
    for name, p in PRESETS.items():
        assert "prompt" in p
        assert "seed" in p
        assert "steps" in p
        assert "scheduler" in p
