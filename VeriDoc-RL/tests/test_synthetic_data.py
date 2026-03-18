from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.data.synthetic import (
    SyntheticFormGenerator,
    build_sft_record,
    build_training_record,
    export_jsonl,
    main,
)


def test_synthetic_generator_is_deterministic() -> None:
    first = SyntheticFormGenerator(seed=11).generate_sample(sample_index=0).to_record()
    second = SyntheticFormGenerator(seed=11).generate_sample(sample_index=0).to_record()

    assert first == second


def test_generated_record_contains_bucket_metadata() -> None:
    sample = SyntheticFormGenerator(seed=5).generate_sample(
        sample_index=3,
        template_family="template_c",
        ocr_noise_level="medium",
        hard_case_type="checkbox_conflict",
        rule_complexity="conditional_checkbox",
    )
    record = build_sft_record(sample)

    assert record["task_type"] == "SFT_gold"
    assert record["metadata"]["bucket"]["template_family"] == "template_c"
    statuses = {item["rule_id"]: item["status"] for item in record["reference"]["validations"]}
    assert statuses["checkbox.payment_mode_exclusive"] == "fail"


def test_export_jsonl_writes_records(tmp_path: Path) -> None:
    records = SyntheticFormGenerator(seed=7).generate_dataset(count=2)
    output_path = tmp_path / "synthetic.jsonl"

    export_jsonl(output_path, records)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert "input" in payload
    assert "reference" in payload


def test_build_training_record_supports_prompt_only_data() -> None:
    sample = SyntheticFormGenerator(seed=7).generate_sample(sample_index=0)

    record = build_training_record(sample, task_type="RL_prompt_only")

    assert record["task_type"] == "RL_prompt_only"
    assert "reference" not in record
    assert "input" in record


def test_cli_main_generates_output_file(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.jsonl"

    exit_code = main([
        "--count",
        "3",
        "--seed",
        "13",
        "--output-path",
        str(output_path),
        "--template-family",
        "template_a",
        "--ocr-noise-level",
        "low",
    ])

    assert exit_code == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["metadata"]["bucket"]["template_family"] == "template_a"
    assert first["metadata"]["bucket"]["ocr_noise_level"] == "low"


def test_cli_main_supports_task_type(tmp_path: Path) -> None:
    output_path = tmp_path / "prompt_only.jsonl"

    exit_code = main(
        [
            "--count",
            "2",
            "--seed",
            "7",
            "--task-type",
            "RL_prompt_only",
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 0
    first = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["task_type"] == "RL_prompt_only"
    assert "reference" not in first
