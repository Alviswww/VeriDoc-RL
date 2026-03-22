from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veridoc_rl.data.preferences import build_preference_pairs, export_preference_jsonl
from veridoc_rl.evaluation.comparison import (
    compare_report_snapshots,
    load_report_snapshot,
    write_comparison_artifacts,
)
from veridoc_rl.evaluation.reporting import evaluate_dataset_files, export_case_records, load_jsonl, write_report
from veridoc_rl.inference.candidates import (
    CandidateGenerationConfig,
    export_candidate_jsonl,
    generate_candidates_for_records,
)
from veridoc_rl.inference.runner import (
    InferenceConfig,
    export_prediction_jsonl,
    run_inference_records,
    select_first_candidate_predictions,
)
from veridoc_rl.orchestration.paths import PipelinePaths
from veridoc_rl.orchestration.spec import PipelineSpec
from veridoc_rl.orchestration.state import PipelineState, mark_stage_running, mark_stage_succeeded
from veridoc_rl.training.corpus import (
    export_training_jsonl,
    prepare_dpo_corpus,
    prepare_rl_corpus,
    prepare_sft_corpus,
)
from veridoc_rl.training.manifests import build_training_manifests, write_training_bundle
from veridoc_rl.training.runtime import (
    build_runtime_launch_plan,
    execute_runtime_plan,
    load_training_manifest,
    write_runtime_bundle,
)


def run_baseline_stage(spec: PipelineSpec, paths: PipelinePaths, state: PipelineState) -> None:
    stage_name = "baseline"
    stage = state.ensure_stage(stage_name)
    if _can_skip(stage, required=paths.stage_report_path(stage_name), resume=spec.execution.resume):
        return

    mark_stage_running(stage, base_model=spec.model.baseline)
    candidate_records = generate_candidates_for_records(
        load_jsonl(Path(spec.data.sft_gold_path)),
        config=CandidateGenerationConfig(
            model=spec.model.baseline,
            backend=spec.model.inference_backend,
            api_base=spec.model.inference_api_base,
            api_key=spec.model.inference_api_key,
            num_candidates=spec.generation.candidate_count,
            temperature=spec.generation.temperature,
            top_p=spec.generation.top_p,
            max_new_tokens=spec.generation.max_new_tokens,
            system_prompt=spec.generation.system_prompt,
            disable_thinking=spec.generation.disable_thinking,
            extra_body=dict(spec.generation.extra_body),
        ),
    )
    candidate_path = paths.stage_candidates_path(stage_name)
    export_candidate_jsonl(candidate_path, candidate_records)
    prediction_rows = select_first_candidate_predictions(candidate_records)
    prediction_path = paths.stage_predictions_path(stage_name)
    export_prediction_jsonl(prediction_path, prediction_rows)
    report = evaluate_dataset_files(
        Path(spec.data.sft_gold_path),
        prediction_path,
    )
    report_path = paths.stage_report_path(stage_name)
    cases_path = paths.stage_cases_path(stage_name)
    write_report(report_path, report)
    export_case_records(cases_path, report.cases, failures_only=False)
    stage.candidate_path = str(candidate_path)
    stage.prediction_path = str(prediction_path)
    stage.report_path = str(report_path)
    stage.case_export_path = str(cases_path)
    mark_stage_succeeded(stage)


def run_sft_stage(spec: PipelineSpec, paths: PipelinePaths, state: PipelineState) -> None:
    stage_name = "phase_a_sft"
    stage = state.ensure_stage(stage_name)
    if _can_skip(
        stage,
        required=_stage_success_artifact(spec, paths, stage_name),
        resume=spec.execution.resume,
    ):
        return

    mark_stage_running(stage, base_model=spec.model.baseline)
    train_path = paths.stage_train_path(stage_name)
    export_training_jsonl(
        train_path,
        prepare_sft_corpus(load_jsonl(Path(spec.data.sft_gold_path))),
    )
    manifest = _prepare_stage_manifest(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        train_path=train_path,
        base_model=spec.model.baseline,
    )
    _prepare_runtime(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        manifest_path=paths.stage_manifest_path(stage_name),
        execute=spec.execution.execute_training and not spec.execution.prepare_only,
    )
    stage.train_data_path = str(train_path)
    stage.manifest_path = str(paths.stage_manifest_path(stage_name))
    stage.runtime_plan_path = str(paths.stage_runtime_plan_path(stage_name))
    stage.checkpoint_path = str(paths.stage_checkpoint_dir(stage_name))
    if spec.execution.execute_training and not spec.execution.prepare_only and spec.pipeline.enable_post_train_eval:
        _run_checkpoint_eval(
            spec=spec,
            paths=paths,
            stage_name=stage_name,
            checkpoint_path=Path(stage.checkpoint_path),
        )
        stage.prediction_path = str(paths.stage_predictions_path(stage_name))
        stage.report_path = str(paths.stage_report_path(stage_name))
        stage.case_export_path = str(paths.stage_cases_path(stage_name))
    del manifest
    mark_stage_succeeded(stage)


def run_dpo_stage(spec: PipelineSpec, paths: PipelinePaths, state: PipelineState) -> None:
    stage_name = "phase_b_dpo"
    stage = state.ensure_stage(stage_name)
    if _can_skip(
        stage,
        required=_stage_success_artifact(spec, paths, stage_name),
        resume=spec.execution.resume,
    ):
        return

    base_model = str(paths.stage_checkpoint_dir("phase_a_sft"))
    mark_stage_running(stage, base_model=base_model)
    candidate_path = _ensure_preference_candidates(spec=spec, paths=paths)
    preferences = build_preference_pairs(
        candidate_records=load_jsonl(candidate_path),
        reference_records=load_jsonl(Path(spec.data.sft_gold_path)),
        min_margin=0.05,
    )
    preferences_path = paths.stage_preferences_path(stage_name)
    export_preference_jsonl(preferences_path, preferences)
    train_path = paths.stage_train_path(stage_name)
    export_training_jsonl(
        train_path,
        prepare_dpo_corpus(load_jsonl(preferences_path)),
    )
    _prepare_stage_manifest(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        train_path=train_path,
        base_model=base_model,
    )
    _prepare_runtime(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        manifest_path=paths.stage_manifest_path(stage_name),
        execute=spec.execution.execute_training and not spec.execution.prepare_only,
    )
    stage.train_data_path = str(train_path)
    stage.candidate_path = str(candidate_path)
    stage.manifest_path = str(paths.stage_manifest_path(stage_name))
    stage.runtime_plan_path = str(paths.stage_runtime_plan_path(stage_name))
    stage.checkpoint_path = str(paths.stage_checkpoint_dir(stage_name))
    if spec.execution.execute_training and not spec.execution.prepare_only and spec.pipeline.enable_post_train_eval:
        _run_checkpoint_eval(
            spec=spec,
            paths=paths,
            stage_name=stage_name,
            checkpoint_path=Path(stage.checkpoint_path),
        )
        stage.prediction_path = str(paths.stage_predictions_path(stage_name))
        stage.report_path = str(paths.stage_report_path(stage_name))
        stage.case_export_path = str(paths.stage_cases_path(stage_name))
    mark_stage_succeeded(stage)


def run_rl_stage(spec: PipelineSpec, paths: PipelinePaths, state: PipelineState) -> None:
    stage_name = f"phase_c_{spec.pipeline.rl_algorithm}"
    stage = state.ensure_stage(stage_name)
    if _can_skip(
        stage,
        required=_stage_success_artifact(spec, paths, stage_name),
        resume=spec.execution.resume,
    ):
        return

    base_model = str(paths.stage_checkpoint_dir("phase_a_sft"))
    mark_stage_running(stage, base_model=base_model)
    train_path = paths.stage_train_path(stage_name)
    export_training_jsonl(
        train_path,
        prepare_rl_corpus(
            load_jsonl(Path(spec.data.rl_prompt_only_path)),
            reward_profile="rlvr",
        ),
    )
    _prepare_stage_manifest(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        train_path=train_path,
        base_model=base_model,
    )
    _prepare_runtime(
        spec=spec,
        paths=paths,
        stage_name=stage_name,
        manifest_path=paths.stage_manifest_path(stage_name),
        execute=spec.execution.execute_training and not spec.execution.prepare_only,
        materialize_data=True,
    )
    stage.train_data_path = str(train_path)
    stage.manifest_path = str(paths.stage_manifest_path(stage_name))
    stage.runtime_plan_path = str(paths.stage_runtime_plan_path(stage_name))
    stage.checkpoint_path = str(paths.stage_checkpoint_dir(stage_name))
    if spec.execution.execute_training and not spec.execution.prepare_only and spec.pipeline.enable_post_train_eval:
        _run_checkpoint_eval(
            spec=spec,
            paths=paths,
            stage_name=stage_name,
            checkpoint_path=Path(stage.checkpoint_path),
        )
        stage.prediction_path = str(paths.stage_predictions_path(stage_name))
        stage.report_path = str(paths.stage_report_path(stage_name))
        stage.case_export_path = str(paths.stage_cases_path(stage_name))
    mark_stage_succeeded(stage)


def write_pipeline_summary(spec: PipelineSpec, paths: PipelinePaths, state: PipelineState) -> None:
    snapshots = []
    for stage_name in _summary_stage_order(spec):
        stage = state.stages.get(stage_name)
        if stage is None or not stage.report_path:
            continue
        report_path = Path(stage.report_path)
        if report_path.exists():
            snapshots.append(load_report_snapshot(stage_name, report_path))

    summary: dict[str, Any] = {
        "run_name": spec.run.name,
        "status": state.status,
        "stages": state.to_dict()["stages"],
    }
    if snapshots:
        comparison = compare_report_snapshots(snapshots)
        artifacts = write_comparison_artifacts(paths.comparison_dir, comparison)
        summary["comparison"] = {
            "reports": comparison["reports"],
            "best_by_metric": comparison["best_by_metric"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }
    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_stage_manifest(
    *,
    spec: PipelineSpec,
    paths: PipelinePaths,
    stage_name: str,
    train_path: Path,
    base_model: str,
) -> Path:
    manifests = build_training_manifests(
        matrix=_load_matrix(spec),
        output_dir=paths.root,
        train_data_paths={stage_name: train_path},
        base_models={stage_name: base_model},
    )
    write_training_bundle(paths.root, manifests)
    return paths.stage_manifest_path(stage_name)


def _prepare_runtime(
    *,
    spec: PipelineSpec,
    paths: PipelinePaths,
    stage_name: str,
    manifest_path: Path,
    execute: bool,
    materialize_data: bool = False,
) -> None:
    manifest = load_training_manifest(manifest_path)
    plan = build_runtime_launch_plan(
        manifest,
        run_dir=paths.stage_dir(stage_name),
        materialize_data=materialize_data or execute,
    )
    write_runtime_bundle(paths.stage_dir(stage_name), plan)
    if execute:
        exit_code = execute_runtime_plan(plan)
        if exit_code != 0:
            raise RuntimeError(f"Training stage {stage_name} failed with exit code {exit_code}.")


def _run_checkpoint_eval(
    *,
    spec: PipelineSpec,
    paths: PipelinePaths,
    stage_name: str,
    checkpoint_path: Path,
) -> None:
    if not checkpoint_path.exists():
        return
    predictions = run_inference_records(
        load_jsonl(Path(spec.data.sft_gold_path)),
        config=InferenceConfig(
            model_name_or_path=str(checkpoint_path),
            system_prompt=spec.generation.system_prompt,
            max_new_tokens=spec.generation.max_new_tokens,
            adapter_config=dict(_load_matrix(spec).finetune),
            precision_config=dict(_load_matrix(spec).finetune.get("precision", {}))
            if isinstance(_load_matrix(spec).finetune.get("precision"), dict)
            else {},
            disable_thinking=spec.generation.disable_thinking,
        ),
    )
    prediction_path = paths.stage_predictions_path(stage_name)
    export_prediction_jsonl(prediction_path, predictions)
    report = evaluate_dataset_files(Path(spec.data.sft_gold_path), prediction_path)
    write_report(paths.stage_report_path(stage_name), report)
    export_case_records(paths.stage_cases_path(stage_name), report.cases, failures_only=False)


def _ensure_preference_candidates(*, spec: PipelineSpec, paths: PipelinePaths) -> Path:
    preference_stage_name = "phase_b_dpo"
    preference_path = paths.stage_candidates_path(preference_stage_name)
    if preference_path.exists():
        return preference_path
    if spec.generation.preference_source == "baseline":
        baseline_path = paths.stage_candidates_path("baseline")
        if baseline_path.exists():
            return baseline_path
        target_model = spec.model.baseline
    else:
        adapter_path = paths.stage_checkpoint_dir("phase_a_sft")
        if not adapter_path.exists():
            raise RuntimeError(
                "DPO preference generation requires a Phase A SFT checkpoint, "
                f"but it was not found at {adapter_path}."
            )
        adapter_name = spec.generation.preference_adapter_name
        target_model = spec.generation.preference_model or f"{spec.model.baseline}:{adapter_name}"
    candidate_records = generate_candidates_for_records(
        load_jsonl(Path(spec.data.sft_gold_path)),
        config=CandidateGenerationConfig(
            model=target_model,
            backend=spec.model.inference_backend,
            api_base=spec.model.inference_api_base,
            api_key=spec.model.inference_api_key,
            num_candidates=spec.generation.candidate_count,
            temperature=spec.generation.temperature,
            top_p=spec.generation.top_p,
            max_new_tokens=spec.generation.max_new_tokens,
            system_prompt=spec.generation.system_prompt,
            disable_thinking=spec.generation.preference_disable_thinking,
            extra_body=dict(spec.generation.preference_extra_body),
        ),
    )
    export_candidate_jsonl(preference_path, candidate_records)
    return preference_path


def _can_skip(stage: Any, *, required: Path, resume: bool) -> bool:
    return bool(resume and stage.status == "succeeded" and required.exists())


def _stage_success_artifact(spec: PipelineSpec, paths: PipelinePaths, stage_name: str) -> Path:
    if spec.execution.execute_training and not spec.execution.prepare_only:
        return paths.stage_report_path(stage_name)
    return paths.stage_runtime_plan_path(stage_name)


def _summary_stage_order(spec: PipelineSpec) -> list[str]:
    names: list[str] = []
    if spec.pipeline.enable_baseline_eval:
        names.append("baseline")
    if spec.pipeline.enable_sft:
        names.append("phase_a_sft")
    if spec.pipeline.enable_dpo:
        names.append("phase_b_dpo")
    if spec.pipeline.enable_rl:
        names.append(f"phase_c_{spec.pipeline.rl_algorithm}")
    return names


def _load_matrix(spec: PipelineSpec) -> Any:
    from veridoc_rl.experiments import load_experiment_matrix

    return load_experiment_matrix(Path(spec.matrix_path))
