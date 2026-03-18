from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.experiments import load_experiment_matrix
from veridoc_rl.training.manifests import build_training_manifests
from veridoc_rl.training.runtime import (
    build_runtime_launch_plan,
    main as prepare_verl_runtime_main,
    write_runtime_bundle,
)
from veridoc_rl.training.verl_reward import compute_score


def test_verl_reward_scores_valid_prediction() -> None:
    reference = {
        "sample_id": "sample-1",
        "fields": {
            "policyholder_name": "张三",
            "policyholder_phone": "13800138000",
        },
        "validations": [],
    }
    prediction = {
        "sample_id": "sample-1",
        "fields": {
            "policyholder_name": "张三",
            "policyholder_phone": "13800138000",
        },
        "validations": [],
    }

    score = compute_score(
        "veridoc_rl",
        json.dumps(prediction, ensure_ascii=False),
        json.dumps(reference, ensure_ascii=False),
        json.dumps({"reward_profile": "rlvr"}, ensure_ascii=False),
    )

    assert score > 0.0


def test_build_runtime_launch_plan_supports_grpo_phase(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        train_data_path=Path("outputs/train.phase_c_rlvr.parquet"),
        output_dir=tmp_path / "bundle",
    )
    manifest = next(item for item in manifests if item.name == "phase_c_grpo")

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.supported is True
    assert "verl.trainer.main_ppo" in plan.command_preview
    assert "algorithm.adv_estimator=grpo" in plan.command_preview
    assert "custom_reward_function.path=" in plan.command_preview


def test_build_runtime_launch_plan_marks_phase_b_unsupported(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        train_data_path=Path("outputs/train.phase_b_dpo.jsonl"),
        output_dir=tmp_path / "bundle",
    )
    manifest = next(item for item in manifests if item.name == "phase_b_dpo")

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.supported is False
    assert "DPO" in plan.reason


def test_prepare_verl_runtime_cli_writes_launch_files(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        train_data_path=Path("outputs/train.phase_c_rlvr.parquet"),
        output_dir=tmp_path / "bundle",
    )
    bundle_dir = tmp_path / "bundle"
    write_runtime_bundle(
        tmp_path / "warmup",
        build_runtime_launch_plan(
            next(item for item in manifests if item.name == "phase_c_grpo"),
            run_dir=tmp_path / "warmup",
        ),
    )
    for manifest in manifests:
        manifest_dir = bundle_dir / manifest.name
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    exit_code = prepare_verl_runtime_main(
        [
            "--manifest-path",
            str(bundle_dir / "phase_c_grpo" / "manifest.json"),
            "--run-dir",
            str(tmp_path / "run_grpo"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "run_grpo" / "runtime_plan.json").exists()
    assert (tmp_path / "run_grpo" / "launch.sh").exists()
