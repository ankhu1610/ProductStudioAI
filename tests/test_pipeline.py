import json
from unittest.mock import MagicMock

from app.core.config import Settings
from app.schemas.generation import GenerationMetadata, GenerationRequest
from app.services.pipeline import ProductStudioPipeline


def test_pipeline_end_to_end_mock(tmp_path, image, product_mask):
    mock_gen_service = MagicMock()
    mock_gen_service.generate_background.return_value = (image("blue", size=(64, 64)), 0.42, 128.0)

    settings = Settings(output_dir=tmp_path, device="cpu")
    pipeline = ProductStudioPipeline(settings=settings, generation_service=mock_gen_service)

    orig_img = image("red", size=(64, 64))
    mask_img = product_mask(size=(64, 64))

    request = GenerationRequest(
        prompt="luxury studio background",
        seed=100,
        steps=10,
        width=64,
        height=64,
        num_variants=2,
        scheduler="pndm",
    )

    result = pipeline.run(orig_img, mask_img, request)

    assert result.request_id is not None
    assert result.output_dir.exists()
    assert len(result.variants) == 2
    assert (result.output_dir / "input_image.png").exists()
    assert (result.output_dir / "input_mask.png").exists()

    v1 = result.variants[0]
    assert v1.final_path.exists()
    assert v1.generated_path.exists()
    assert v1.metadata_path.exists()
    assert v1.metadata.foreground_pixel_difference == 0.0
    assert v1.metadata.peak_gpu_memory_mb == 128.0
    assert v1.metadata.seed == 100
    assert v1.metadata.scheduler == "pndm"

    v2 = result.variants[1]
    assert v2.metadata.seed == 101

    with open(v1.metadata_path, encoding="utf-8") as handle:
        meta_dict = json.load(handle)
    validated_metadata = GenerationMetadata.model_validate(meta_dict)
    assert validated_metadata.prompt == "luxury studio background"
    assert validated_metadata.steps == 10
    assert validated_metadata.foreground_pixel_difference == 0.0
