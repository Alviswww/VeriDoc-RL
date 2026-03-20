from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.orchestration.paths import PipelinePaths
from veridoc_rl.orchestration.runner import run_pipeline
from veridoc_rl.orchestration.spec import load_pipeline_spec
from veridoc_rl.orchestration.state import (
    PipelineState,
    load_or_create_state,
    mark_stage_running,
    mark_stage_succeeded,
    save_state,
)


def test_load_pipeline_spec_from_yaml(tmp_path: Path) -> None:
    spec_path = tmp_path / "pipeline.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "run:",
                '  name: "demo_run"',
                '  output_root: "outputs/pipelines"',
                "model:",
                '  baseline: "models/Qwen3-0.6B"',
                "data:",
                '  sft_gold_path: "outputs/sft_gold.jsonl"',
                '  rl_prompt_only_path: "outputs/rl_prompt_only.jsonl"',
                "pipeline:",
                "  enable_rl: false",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_pipeline_spec(spec_path)

    assert spec.run.name == "demo_run"
    assert spec.model.baseline == "models/Qwen3-0.6B"
    assert spec.pipeline.enable_rl is False
    assert spec.generation.candidate_count == 4


def test_load_pipeline_spec_expands_env_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VERIDOC_OUTPUT_ROOT", str(tmp_path / "pipelines"))
    monkeypatch.setenv("VERIDOC_MODEL_PATH", str(tmp_path / "models" / "Qwen3-0.6B"))
    monkeypatch.setenv("VERIDOC_SFT_GOLD_PATH", str(tmp_path / "outputs" / "sft_gold.jsonl"))
    monkeypatch.setenv(
        "VERIDOC_RL_PROMPT_ONLY_PATH",
        str(tmp_path / "outputs" / "rl_prompt_only.jsonl"),
    )
    monkeypatch.setenv("VERIDOC_API_BASE", "http://127.0.0.1:30000/v1")

    spec_path = tmp_path / "pipeline.env.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "run:",
                '  name: "demo_run"',
                '  output_root: "${VERIDOC_OUTPUT_ROOT}"',
                "model:",
                '  baseline: "${VERIDOC_MODEL_PATH}"',
                '  inference_api_base: "${VERIDOC_API_BASE}"',
                "data:",
                '  sft_gold_path: "${VERIDOC_SFT_GOLD_PATH}"',
                '  rl_prompt_only_path: "${VERIDOC_RL_PROMPT_ONLY_PATH}"',
            ]
        ),
        encoding="utf-8",
    )

    spec = load_pipeline_spec(spec_path)

    assert spec.run.output_root == str(tmp_path / "pipelines")
    assert spec.model.baseline == str(tmp_path / "models" / "Qwen3-0.6B")
    assert spec.data.sft_gold_path == str(tmp_path / "outputs" / "sft_gold.jsonl")


def test_pipeline_state_roundtrip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = PipelineState(run_name="demo")
    stage = state.ensure_stage("phase_a_sft")
    mark_stage_running(stage, base_model="models/Qwen3-0.6B")
    stage.manifest_path = "manifest.json"
    mark_stage_succeeded(stage)
    save_state(state_path, state)

    loaded = load_or_create_state(state_path, run_name="demo", stage_names=["phase_a_sft"])

    assert loaded.run_name == "demo"
    assert loaded.stages["phase_a_sft"].status == "succeeded"
    assert loaded.stages["phase_a_sft"].manifest_path == "manifest.json"


def test_pipeline_paths_layout() -> None:
    spec = load_pipeline_spec(Path("configs/pipeline.qwen3_0p6.yaml"))
    paths = PipelinePaths.from_spec(spec)

    assert str(paths.stage_dir("phase_a_sft")).endswith("qwen3_0p6_mainline/phase_a_sft")
    assert str(paths.stage_manifest_path("phase_b_dpo")).endswith("phase_b_dpo/manifest.json")
    assert str(paths.stage_checkpoint_dir("phase_c_grpo")).endswith("phase_c_grpo/checkpoints")


def test_run_pipeline_uses_stubbed_stage_executors(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "pipeline.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "run:",
                '  name: "demo_run"',
                f'  output_root: "{tmp_path.as_posix()}"',
                "model:",
                '  baseline: "models/Qwen3-0.6B"',
                "data:",
                f'  sft_gold_path: "{(tmp_path / "sft_gold.jsonl").as_posix()}"',
                f'  rl_prompt_only_path: "{(tmp_path / "rl_prompt_only.jsonl").as_posix()}"',
                "pipeline:",
                "  enable_baseline_eval: true",
                "  enable_sft: true",
                "  enable_dpo: true",
                "  enable_rl: true",
                '  rl_algorithm: "grpo"',
                "execution:",
                "  prepare_only: true",
                "  execute_training: false",
                "  resume: true",
            ]
        ),
        encoding="utf-8",
    )
    for filename in ("sft_gold.jsonl", "rl_prompt_only.jsonl"):
        (tmp_path / filename).write_text("{}\n", encoding="utf-8")

    def _stub(stage_name: str):
        def runner(spec, paths, state):  # type: ignore[no-untyped-def]
            stage = state.ensure_stage(stage_name)
            stage.status = "succeeded"
            stage.manifest_path = str(paths.stage_manifest_path(stage_name))
            paths.stage_dir(stage_name).mkdir(parents=True, exist_ok=True)
            if stage_name != "baseline":
                paths.stage_manifest_path(stage_name).write_text("{}", encoding="utf-8")
        return runner

    monkeypatch.setattr("veridoc_rl.orchestration.stages.run_baseline_stage", _stub("baseline"))
    monkeypatch.setattr("veridoc_rl.orchestration.stages.run_sft_stage", _stub("phase_a_sft"))
    monkeypatch.setattr("veridoc_rl.orchestration.stages.run_dpo_stage", _stub("phase_b_dpo"))
    monkeypatch.setattr("veridoc_rl.orchestration.stages.run_rl_stage", _stub("phase_c_grpo"))

    state = run_pipeline(load_pipeline_spec(spec_path))

    assert state.status == "succeeded"
    assert (tmp_path / "demo_run" / "state.json").exists()
    assert (tmp_path / "demo_run" / "summary.json").exists()
    payload = json.loads((tmp_path / "demo_run" / "state.json").read_text(encoding="utf-8"))
    assert payload["stages"]["phase_a_sft"]["status"] == "succeeded"
    assert payload["stages"]["phase_c_grpo"]["status"] == "succeeded"
