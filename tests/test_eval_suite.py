import json

from eval.eval_suite import BENCHMARK_SPEC_PATH, load_benchmark_spec, run_benchmark_suite


def test_load_benchmark_spec_has_ten_cases():
    defaults, cases = load_benchmark_spec(BENCHMARK_SPEC_PATH)
    assert defaults["width"] == 512
    assert len(cases) == 10
    assert cases[0].id == "sneaker-neon"
    assert cases[0].image.exists()
    assert cases[0].mask.exists()


def test_run_benchmark_suite_mock(tmp_path, image, product_mask, monkeypatch):
    from app.core.config import Settings
    from app.services.pipeline import ProductStudioPipeline, VariantResult
    from app.schemas.generation import GenerationMetadata

    defaults, cases = load_benchmark_spec(BENCHMARK_SPEC_PATH)
    case = cases[0]

    metadata = GenerationMetadata(
        request_id="test_job",
        created_at="2026-08-14T00:00:00+00:00",
        base_model="test-model",
        scheduler="ddim",
        seed=case.seed,
        steps=10,
        guidance_scale=7.5,
        prompt=case.prompt,
        negative_prompt="",
        resolution="512x512",
        latency_seconds=1.25,
        peak_gpu_memory_mb=4096.0,
        device="cpu",
        foreground_pixel_difference=0.0,
        input_image="input.png",
        input_mask="mask.png",
        generated_image="generated.png",
        final_image="final.png",
    )
    variant = VariantResult(
        final_image=image("blue", size=(64, 64)),
        generated_background=image("green", size=(64, 64)),
        metadata=metadata,
        final_path=tmp_path / "final.png",
        generated_path=tmp_path / "generated.png",
        metadata_path=tmp_path / "metadata.json",
    )

    class FakePipeline:
        def run(self, image_input, mask_input, request, output_dir=None):
            del image_input, mask_input, request, output_dir
            return type(
                "PipelineResult",
                (),
                {
                    "request_id": "test_job",
                    "output_dir": tmp_path,
                    "variants": [variant],
                },
            )()

    monkeypatch.setattr("eval.eval_suite.ProductStudioPipeline", lambda settings: FakePipeline())
    monkeypatch.setattr("eval.eval_suite.clip_similarity", lambda image, prompt: 0.42)

    benchmark_dir = run_benchmark_suite(
        settings=Settings(output_dir=tmp_path, device="cpu"),
        output_dir=tmp_path / "benchmark",
        step_counts=(10,),
        schedulers=("ddim",),
        compute_clip=True,
        log_mlflow=False,
    )

    results_csv = benchmark_dir / "benchmark_results.csv"
    summary_json = benchmark_dir / "benchmark_summary.json"
    assert results_csv.exists()
    assert summary_json.exists()

    with open(summary_json, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["successful_runs"] == 10
    assert summary["recommendation"]["scheduler"] == "ddim"
    assert summary["recommendation"]["steps"] == 10
