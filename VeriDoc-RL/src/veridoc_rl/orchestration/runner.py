from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

from veridoc_rl.orchestration.paths import PipelinePaths
from veridoc_rl.orchestration.spec import PipelineSpec, load_pipeline_spec
from veridoc_rl.orchestration.state import (
    PipelineState,
    load_or_create_state,
    mark_stage_failed,
    save_state,
)
from veridoc_rl.orchestration import stages


def run_pipeline(spec: PipelineSpec) -> PipelineState:
    paths = PipelinePaths.from_spec(spec)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.spec_snapshot_path.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = load_or_create_state(
        paths.state_path,
        run_name=spec.run.name,
        stage_names=_build_stage_order(spec),
    )
    state.status = "running"
    save_state(paths.state_path, state)

    try:
        for stage_name in _build_stage_order(spec):
            executor = _resolve_stage_executor(stage_name)
            executor(spec, paths, state)
            save_state(paths.state_path, state)
        state.status = "succeeded"
    except Exception as exc:
        failing_stage = _current_running_stage(state) or _build_stage_order(spec)[0]
        mark_stage_failed(state.ensure_stage(failing_stage), str(exc))
        state.status = "failed"
        save_state(paths.state_path, state)
        raise

    stages.write_pipeline_summary(spec, paths, state)
    save_state(paths.state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VeriDoc-RL multi-stage pipeline.")
    parser.add_argument("--spec-path", type=Path, required=True, help="Pipeline spec YAML/JSON path.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare manifests/runtime only and skip actual training execution.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing stage state and rerun every stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = load_pipeline_spec(args.spec_path)
    if args.prepare_only:
        spec = replace(
            spec,
            execution=replace(spec.execution, prepare_only=True, execute_training=False),
        )
    if args.no_resume:
        spec = replace(spec, execution=replace(spec.execution, resume=False))
    run_pipeline(spec)
    return 0


def _build_stage_order(spec: PipelineSpec) -> list[str]:
    stage_order: list[str] = []
    if spec.pipeline.enable_baseline_eval:
        stage_order.append("baseline")
    if spec.pipeline.enable_sft:
        stage_order.append("phase_a_sft")
    if spec.pipeline.enable_dpo:
        stage_order.append("phase_b_dpo")
    if spec.pipeline.enable_rl:
        stage_order.append(f"phase_c_{spec.pipeline.rl_algorithm}")
    return stage_order


def _resolve_stage_executor(stage_name: str) -> Callable[[PipelineSpec, PipelinePaths, PipelineState], None]:
    mapping = {
        "baseline": stages.run_baseline_stage,
        "phase_a_sft": stages.run_sft_stage,
        "phase_b_dpo": stages.run_dpo_stage,
        "phase_c_grpo": stages.run_rl_stage,
        "phase_c_rloo": stages.run_rl_stage,
    }
    if stage_name not in mapping:
        raise ValueError(f"Unsupported pipeline stage: {stage_name}")
    return mapping[stage_name]


def _current_running_stage(state: PipelineState) -> str | None:
    for stage_name, stage in state.stages.items():
        if stage.status == "running":
            return stage_name
    return None


if __name__ == "__main__":
    raise SystemExit(main())
