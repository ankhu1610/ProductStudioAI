import csv
import json
from unittest.mock import MagicMock
from PIL import Image
import pytest

from app.core.config import Settings
from eval.runner import BenchmarkRunner


def test_benchmark_runner_mock(tmp_path):
    mock_gen_service = MagicMock()
    # Mock generation service returns (Image, latency, peak_gpu_vram)
    mock_gen_service.generate_background.return_value = (
        Image.new("RGB", (512, 512), color="blue"),
        0.25,
        1500.0,
    )

    settings = Settings(device="cpu", output_dir=tmp_path)
    runner = BenchmarkRunner(
        settings=settings,
        generation_service=mock_gen_service,
        spec_path="eval/benchmark_prompts.json",
    )

    # Test only 2 cases with 2 step settings on 1 scheduler for fast mock test
    res = runner.run_benchmark(
        steps=[10, 15],
        schedulers=["ddim"],
        case_ids=["sneaker-neon", "watch-marble"],
        skip_clip=True,
        output_dir=tmp_path / "benchmarks",
        log_mlflow=False,
    )

    # Check files created
    assert (tmp_path / "benchmarks" / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmarks" / "benchmark_summary.json").exists()
    assert (tmp_path / "benchmarks" / "human_rating_sheet.csv").exists()

    # Verify CSV rows (header + 2 cases * 2 steps = 5 rows)
    with open(res["results_csv"], encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 5  # header + 4 rows
        assert reader[0][0] == "case_id"
        assert reader[1][0] in ["sneaker-neon", "watch-marble"]

    # Verify summary JSON
    with open(res["summary_json"], encoding="utf-8") as f:
        summary_data = json.load(f)
        assert "configurations" in summary_data
        assert "ddim_10steps" in summary_data["configurations"]
        assert "ddim_15steps" in summary_data["configurations"]
        assert "recommended_configuration" in summary_data
        assert summary_data["configurations"]["ddim_10steps"]["total_runs"] == 2
