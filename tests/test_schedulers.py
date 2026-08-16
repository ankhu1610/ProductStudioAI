from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.generation import GenerationRequest
from app.services.generation import GenerationService


def test_generation_request_schedulers():
    for valid_sched in ["ddim", "pndm", "dpm_solver", "euler_a"]:
        req = GenerationRequest(prompt="studio background", scheduler=valid_sched)
        assert req.scheduler == valid_sched

    with pytest.raises(ValidationError):
        GenerationRequest(prompt="studio background", scheduler="invalid_scheduler")  # type: ignore[arg-type]


def test_generation_service_apply_scheduler():
    settings = Settings(device="cpu")
    service = GenerationService(settings)

    mock_pipeline = MagicMock()
    mock_pipeline.scheduler.config = {"num_train_timesteps": 1000}
    service._pipeline = mock_pipeline

    for sched in ["ddim", "pndm", "dpm_solver", "euler_a"]:
        service._apply_scheduler(sched)
        assert service._active_scheduler == sched
        assert service._pipeline.scheduler is not None

    with pytest.raises(ValueError, match="unsupported scheduler"):
        service._apply_scheduler("unsupported")  # type: ignore[arg-type]
