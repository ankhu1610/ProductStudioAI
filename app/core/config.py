import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PRODUCTSTUDIO_", extra="ignore")

    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    model_path: Path | None = None
    device: str = "cuda"
    precision: Literal["fp16", "fp32"] = "fp16"

    output_dir: Path = Path("outputs")
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 15
    default_guidance_scale: float = 7.5
    max_variants: int = 4

    database_url: str = "sqlite:///productstudio.db"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    @field_validator("default_width", "default_height")
    @classmethod
    def dimensions_must_be_divisible_by_eight(cls, value: int) -> int:
        if value < 64 or value % 8 != 0:
            raise ValueError("image dimensions must be at least 64 and divisible by 8")
        return value

    @field_validator("default_steps", "max_variants")
    @classmethod
    def positive_integers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be at least 1")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return one cached, validated settings instance per process."""

    return Settings()
