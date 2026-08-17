from app.services.database import DatabaseService, JobRecord


def test_database_save_and_get(tmp_path):
    db_file = tmp_path / "test.db"
    db = DatabaseService(db_path=db_file)

    record = JobRecord(
        job_id="job_123",
        created_at="2026-08-16T12:00:00+00:00",
        status="completed",
        prompt="luxury watch on marble",
        negative_prompt="blurry",
        scheduler="ddim",
        steps=15,
        guidance_scale=7.5,
        seed=42,
        resolution="512x512",
        num_variants=1,
        latency_seconds=1.85,
        peak_gpu_memory_mb=2048.0,
        foreground_pixel_difference=0.0,
        clip_similarity=0.34,
        output_dir=str(tmp_path / "job_123"),
        final_images=["final_variant_1.png"],
        metadata_files=["metadata_variant_1.json"],
    )

    db.save_job(record)

    fetched = db.get_job("job_123")
    assert fetched is not None
    assert fetched.job_id == "job_123"
    assert fetched.status == "completed"
    assert fetched.prompt == "luxury watch on marble"
    assert fetched.steps == 15
    assert fetched.final_images == ["final_variant_1.png"]
    assert fetched.foreground_pixel_difference == 0.0


def test_database_list_jobs(tmp_path):
    db_file = tmp_path / "test.db"
    db = DatabaseService(db_path=db_file)

    for i in range(5):
        record = JobRecord(
            job_id=f"job_{i}",
            created_at=f"2026-08-16T12:0{i}:00+00:00",
            status="completed",
            prompt=f"prompt {i}",
            negative_prompt="",
            scheduler="ddim",
            steps=15,
            guidance_scale=7.5,
            seed=i,
            resolution="512x512",
            num_variants=1,
        )
        db.save_job(record)

    jobs = db.list_jobs(limit=3, offset=0)
    assert len(jobs) == 3
    # Order by created_at DESC
    assert jobs[0].job_id == "job_4"
