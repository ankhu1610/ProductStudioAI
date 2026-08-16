import pytest
from pydantic import ValidationError

from app.schemas.generation import GenerationRequest


def test_generation_request_defaults_are_valid():
    request = GenerationRequest(prompt="premium studio background")
    assert request.width == 512
    assert request.scheduler == "ddim"


def test_generation_request_accepts_pndm_scheduler():
    request = GenerationRequest(prompt="premium studio background", scheduler="pndm")
    assert request.scheduler == "pndm"


def test_generation_request_rejects_invalid_resolution():
    with pytest.raises(ValidationError, match="divisible by 8"):
        GenerationRequest(prompt="premium studio background", width=513)

