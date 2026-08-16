from unittest.mock import MagicMock, patch
from PIL import Image
import torch

from app.services.metrics import (
    compute_clip_similarity,
    compute_percentiles,
    get_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
)


def test_compute_percentiles():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    stats = compute_percentiles(data)
    assert stats["p50"] == 5.5
    assert stats["min"] == 1.0
    assert stats["max"] == 10.0
    assert stats["mean"] == 5.5
    assert 9.0 <= stats["p95"] <= 10.0

    empty_stats = compute_percentiles([])
    assert empty_stats["p50"] == 0.0
    assert empty_stats["mean"] == 0.0


def test_gpu_memory_tracking_safe():
    reset_peak_gpu_memory()
    mem = get_peak_gpu_memory_mb()
    assert mem is None or isinstance(mem, float)


@patch("app.services.metrics.get_clip_pipeline")
def test_compute_clip_similarity_mock(mock_get_clip):
    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_get_clip.return_value = (mock_model, mock_processor)
    mock_processor.return_value = {
        "pixel_values": torch.zeros((1, 3, 224, 224)),
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
    }
    mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])

    img = Image.new("RGB", (64, 64), color="red")
    sim = compute_clip_similarity(img, "a red image", device="cpu")
    assert sim == 1.0
