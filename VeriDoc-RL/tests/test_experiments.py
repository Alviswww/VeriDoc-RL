from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.evaluation.comparison import (
    compare_report_snapshots,
    load_report_snapshot,
    main as comparison_main,
)
from veridoc_rl.experiments.matrix import (
    build_experiment_plan,
    load_experiment_matrix,
    main as experiment_plan_main,
)


def test_load_experiment_matrix_and_expand_plan() -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    plan = build_experiment_plan(matrix)

    assert matrix.project["name"] == "VeriDoc-RL"
    assert len(matrix.training_stages) == 3
    assert matrix.finetune["adapter_type"] == "qlora"
    assert matrix.inference["backend"] == "sglang"
    assert [item["experiment_name"] for item in plan] == [
        "sft_only",
        "sft_plus_dpo",
        "sft_plus_rlvr",
        "sft_plus_rlvr_without_cross_field_consistency",
        "sft_plus_rlvr_without_checkbox_logic",
    ]
    assert plan[0]["recommended_model"] == "models/Qwen3-0.6B"
    assert plan[2]["reward_profile"] == "rlvr"
    assert plan[3]["reward_profile"] == "rlvr_without_cross_field_consistency"


def test_experiment_plan_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "plan.json"
    markdown_path = tmp_path / "plan.md"

    exit_code = experiment_plan_main(
        [
            "--matrix-path",
            "configs/experiment_matrix.yaml",
            "--output-path",
            str(output_path),
            "--markdown-path",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"]) == 5
    assert "Recommended Experiment Queue" in markdown_path.read_text(encoding="utf-8")


def test_compare_reports_builds_bucket_summary(tmp_path: Path) -> None:
    snapshots = [
        load_report_snapshot("sft", _write_report_fixture(tmp_path / "sft.json", 0.70, 0.60)),
        load_report_snapshot("rlvr", _write_report_fixture(tmp_path / "rlvr.json", 0.82, 0.78)),
    ]

    comparison = compare_report_snapshots(snapshots)

    assert comparison["best_by_metric"]["rule_pass_rate"]["label"] == "rlvr"
    assert comparison["bucket_comparison"]["high"]["rlvr"] == 0.78


def test_compare_reports_cli_writes_artifacts(tmp_path: Path) -> None:
    sft_path = _write_report_fixture(tmp_path / "sft.json", 0.72, 0.61)
    rlvr_path = _write_report_fixture(tmp_path / "rlvr.json", 0.86, 0.80)
    output_dir = tmp_path / "comparison"

    exit_code = comparison_main(
        [
            "--report",
            f"sft={sft_path}",
            "--report",
            f"rlvr={rlvr_path}",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.md").exists()
    assert (output_dir / "rule_pass_rate_comparison.svg").exists()
    assert (output_dir / "ocr_noise_level_field_f1.svg").exists()


def _write_report_fixture(path: Path, rule_pass_rate: float, high_bucket_field_f1: float) -> Path:
    payload = {
        "overall": {
            "sample_count": 4,
            "failure_count": 1,
            "field_f1": rule_pass_rate - 0.1,
            "form_exact_match": rule_pass_rate - 0.2,
            "rule_pass_rate": rule_pass_rate,
            "validation_match_rate": rule_pass_rate - 0.05,
            "total_reward": rule_pass_rate - 0.15,
        },
        "bucket_metrics": {
            "ocr_noise_level": {
                "low": {"field_f1": min(high_bucket_field_f1 + 0.15, 1.0)},
                "high": {"field_f1": high_bucket_field_f1},
            }
        },
        "failure_cases": [
            {
                "sample_id": "sample-1",
                "taxonomy": ["missing_field", "checkbox_logic_error"],
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
