from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.experiments import load_experiment_matrix
from veridoc_rl.training.manifests import build_training_manifests
from veridoc_rl.training.runtime import (
    build_runtime_launch_plan,
    execute_runtime_plan,
    write_runtime_bundle,
)
from veridoc_rl.training.runtime import (
    main as prepare_verl_runtime_main,
)
from veridoc_rl.training.verl_reward import compute_score


def _write_dpo_corpus(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "task_type": "DPO_preference",
                "stage": "phase_b_dpo",
                "sample_id": "sample-pref",
                "system_prompt": "Return JSON only.",
                "prompt": "Extract the fields.",
                "chosen": (
                    '{"sample_id":"sample-pref","fields":{"policyholder_name":"张三"},'
                    '"validations":[]}'
                ),
                "rejected": (
                    '{"sample_id":"sample-pref","fields":{"policyholder_name":""},'
                    '"validations":[]}'
                ),
                "reward_profile": "default",
                "reward_margin": 0.2,
                "chosen_candidate_id": "good",
                "rejected_candidate_id": "bad",
                "metadata": {"bucket": {"template_family": "template_a"}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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
        output_dir=tmp_path / "bundle",
        train_data_paths={
            "phase_c_grpo": Path("outputs/train.phase_c_rlvr.parquet"),
            "phase_c_rloo": Path("outputs/train.phase_c_rlvr.parquet"),
        },
    )
    manifest = next(item for item in manifests if item.name == "phase_c_grpo")

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.supported is True
    assert plan.runtime_backend == "verl"
    assert "verl.trainer.main_ppo" in plan.command_preview
    assert "algorithm.adv_estimator=grpo" in plan.command_preview
    assert "custom_reward_function.path=" in plan.command_preview


def test_build_runtime_launch_plan_uses_rl_python_bin_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={"phase_c_grpo": Path("outputs/train.phase_c_rlvr.parquet")},
    )
    manifest = next(item for item in manifests if item.name == "phase_c_grpo")
    rl_python = tmp_path / "envs" / "rl" / "bin" / "python"
    rl_python.parent.mkdir(parents=True, exist_ok=True)
    rl_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("VERIDOC_RL_PYTHON_BIN", str(rl_python))

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.command[0] == str(rl_python)


def test_build_runtime_launch_plan_supports_phase_b_dpo(tmp_path: Path) -> None:
    dpo_corpus_path = _write_dpo_corpus(tmp_path / "train.phase_b_dpo.jsonl")
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={"phase_b_dpo": dpo_corpus_path},
    )
    manifest = next(item for item in manifests if item.name == "phase_b_dpo")

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.supported is True
    assert plan.runtime_backend == "trl"
    assert "veridoc_rl.training.trl_dpo" in plan.command_preview
    assert (tmp_path / "runs" / manifest.name / "dpo_config.json").exists()
    assert (tmp_path / "runs" / manifest.name / "data" / "phase_b_dpo.train.jsonl").exists()


def test_build_runtime_launch_plan_supports_phase_a_sft(tmp_path: Path) -> None:
    sft_corpus_path = tmp_path / "train.phase_a_sft.jsonl"
    sft_corpus_path.write_text(
        json.dumps(
            {
                "task_type": "SFT_gold",
                "stage": "phase_a_sft",
                "sample_id": "sample-1",
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": "Extract the fields."},
                    {"role": "assistant", "content": '{"sample_id":"sample-1","fields":{},"validations":[]}'},
                ],
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={"phase_a_sft": sft_corpus_path},
    )
    manifest = next(item for item in manifests if item.name == "phase_a_sft")

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.supported is True
    assert plan.runtime_backend == "transformers"
    assert "veridoc_rl.training.trl_sft" in plan.command_preview
    assert (tmp_path / "runs" / manifest.name / "sft_config.json").exists()
    assert (tmp_path / "runs" / manifest.name / "data" / "phase_a_sft.train.jsonl").exists()


def test_build_runtime_launch_plan_uses_train_python_bin_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sft_corpus_path = tmp_path / "train.phase_a_sft.jsonl"
    sft_corpus_path.write_text(
        json.dumps(
            {
                "task_type": "SFT_gold",
                "stage": "phase_a_sft",
                "sample_id": "sample-1",
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": "Extract the fields."},
                    {
                        "role": "assistant",
                        "content": '{"sample_id":"sample-1","fields":{},"validations":[]}',
                    },
                ],
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={"phase_a_sft": sft_corpus_path},
    )
    manifest = next(item for item in manifests if item.name == "phase_a_sft")
    train_python = tmp_path / "envs" / "train" / "bin" / "python"
    train_python.parent.mkdir(parents=True, exist_ok=True)
    train_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("VERIDOC_TRAIN_PYTHON_BIN", str(train_python))

    plan = build_runtime_launch_plan(manifest, run_dir=tmp_path / "runs" / manifest.name)

    assert plan.command[0] == str(train_python)


def test_prepare_verl_runtime_cli_writes_launch_files(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={
            "phase_c_grpo": Path("outputs/train.phase_c_rlvr.parquet"),
            "phase_c_rloo": Path("outputs/train.phase_c_rlvr.parquet"),
        },
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


def test_execute_runtime_plan_uses_target_launcher(monkeypatch) -> None:
    from veridoc_rl.training.runtime import RuntimeLaunchPlan

    captured: dict[str, object] = {}

    def _fake_run(command, check):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["check"] = check

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    plan = RuntimeLaunchPlan(
        supported=True,
        manifest_name="phase_c_grpo",
        phase="phase_c_grpo",
        runtime_backend="verl",
        run_dir=".",
        command=["/bin/true"],
        command_preview="/bin/true",
        train_data_path=None,
        eval_data_path=None,
        generated_files=[],
        reason="",
    )

    exit_code = execute_runtime_plan(plan)

    assert exit_code == 0
    assert captured["command"] == ["/bin/true"]
    assert captured["check"] is False


def test_prepare_runtime_cli_writes_dpo_launch_files(tmp_path: Path) -> None:
    dpo_corpus_path = _write_dpo_corpus(tmp_path / "train.phase_b_dpo.jsonl")
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))
    manifests = build_training_manifests(
        matrix,
        output_dir=tmp_path / "bundle",
        train_data_paths={"phase_b_dpo": dpo_corpus_path},
    )
    manifest = next(item for item in manifests if item.name == "phase_b_dpo")
    manifest_dir = tmp_path / "bundle" / manifest.name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    exit_code = prepare_verl_runtime_main(
        [
            "--manifest-path",
            str(manifest_dir / "manifest.json"),
            "--run-dir",
            str(tmp_path / "run_dpo"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "run_dpo" / "runtime_plan.json").exists()
    assert (tmp_path / "run_dpo" / "launch.sh").exists()
    assert (tmp_path / "run_dpo" / "dpo_config.json").exists()


def test_load_training_manifest_expands_env_vars(tmp_path: Path, monkeypatch) -> None:
    from veridoc_rl.training.runtime import load_training_manifest

    monkeypatch.setenv("VERIDOC_MODEL_REF", str(tmp_path / "models" / "Qwen3-1.7B"))
    monkeypatch.setenv("VERIDOC_TRAIN_PATH", str(tmp_path / "train.jsonl"))
    monkeypatch.setenv("VERIDOC_OUTPUT_PATH", str(tmp_path / "outputs" / "phase_a_sft"))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "phase_a_sft",
                "phase": "phase_a_sft",
                "backend": "multi",
                "algorithm": "sft",
                "base_model": "${VERIDOC_MODEL_REF}",
                "base_model_source": "baseline",
                "train_data_path": "${VERIDOC_TRAIN_PATH}",
                "eval_data_path": None,
                "output_dir": "${VERIDOC_OUTPUT_PATH}",
                "prompt_template": "veridoc_v1",
                "reward_profile": "default",
                "adapter_config": {},
                "precision_config": {},
                "trainer": {},
                "runtime": {},
                "notes": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = load_training_manifest(manifest_path)

    assert manifest.base_model == str(tmp_path / "models" / "Qwen3-1.7B")
    assert manifest.train_data_path == str(tmp_path / "train.jsonl")
    assert manifest.output_dir == str(tmp_path / "outputs" / "phase_a_sft")
