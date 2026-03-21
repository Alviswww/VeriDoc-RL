from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.experiments import ExperimentMatrix, load_experiment_matrix
from veridoc_rl.model_defaults import DEFAULT_BASELINE_MODEL

DEFAULT_RUNTIME = {
    "backend": "multi",
    "prompt_template": "veridoc_v1",
    "phases": {
        "phase_a_sft": {
            "algorithm": "sft",
            "reward_profile": "default",
            "epochs": 1,
            "learning_rate": 0.0002,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_length": 3072,
            "logging_steps": 1,
            "save_steps": 10,
        },
        "phase_b_dpo": {
            "algorithm": "dpo",
            "reward_profile": "default",
            "epochs": 1,
            "learning_rate": 0.000001,
            "beta": 0.1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_length": 3072,
            "max_prompt_length": 2048,
            "max_completion_length": 1024,
            "logging_steps": 1,
            "save_steps": 10,
        },
        "phase_c_grpo": {
            "algorithm": "grpo",
            "reward_profile": "rlvr",
            "epochs": 1,
            "learning_rate": 0.000001,
            "rollout_n": 4,
        },
        "phase_c_rloo": {
            "algorithm": "rloo",
            "reward_profile": "rlvr",
            "epochs": 1,
            "learning_rate": 0.000001,
            "rollout_n": 4,
        },
    },
}


@dataclass(slots=True, frozen=True)
class TrainingManifest:
    name: str
    phase: str
    backend: str
    algorithm: str
    base_model: str
    base_model_source: str
    train_data_path: str
    eval_data_path: str | None
    output_dir: str
    prompt_template: str
    reward_profile: str
    adapter_config: dict[str, Any]
    precision_config: dict[str, Any]
    trainer: dict[str, Any]
    runtime: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "base_model": self.base_model,
            "base_model_source": self.base_model_source,
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            "output_dir": self.output_dir,
            "prompt_template": self.prompt_template,
            "reward_profile": self.reward_profile,
            "adapter_config": dict(self.adapter_config),
            "precision_config": dict(self.precision_config),
            "trainer": dict(self.trainer),
            "runtime": dict(self.runtime),
            "notes": list(self.notes),
        }


def build_training_manifests(
    matrix: ExperimentMatrix,
    *,
    output_dir: Path,
    train_data_path: Path | None = None,
    train_data_paths: Mapping[str, Path] | None = None,
    eval_data_path: Path | None = None,
    eval_data_paths: Mapping[str, Path] | None = None,
    base_model: str | None = None,
    base_models: Mapping[str, str] | None = None,
) -> list[TrainingManifest]:
    runtime = _resolve_runtime_config(matrix)
    backend = str(runtime.get("backend", "verl"))
    prompt_template = str(runtime.get("prompt_template", "veridoc_v1"))
    phase_configs = _as_mapping(runtime.get("phases"), field_name="training_runtime.phases")
    common_runtime = _as_mapping(runtime.get("common", {}), field_name="training_runtime.common")

    manifests: list[TrainingManifest] = []
    for phase_name, config in phase_configs.items():
        phase_config = _as_mapping(config, field_name=f"training_runtime.phases.{phase_name}")
        phase_train_data_path = _resolve_phase_path(
            phase_name,
            legacy_path=train_data_path,
            phase_paths=train_data_paths,
        )
        if phase_train_data_path is None:
            continue
        phase_eval_data_path = _resolve_phase_path(
            phase_name,
            legacy_path=eval_data_path,
            phase_paths=eval_data_paths,
        )
        manifest_base_model = _resolve_phase_base_model(
            phase_name,
            matrix=matrix,
            legacy_model=base_model,
            phase_models=base_models,
        )
        reward_profile = str(phase_config.get("reward_profile", "default"))
        manifest_output_dir = output_dir / phase_name
        notes = [
            (
                "The same prompt template and verifier reward profile should be reused "
                "across train/eval/reporting."
            ),
        ]
        if phase_name != "phase_a_sft":
            notes.append(
                "For DPO and RLVR runs, prefer passing the SFT checkpoint path as the base model."
            )
        manifests.append(
            TrainingManifest(
                name=phase_name,
                phase=phase_name,
                backend=backend,
                algorithm=str(phase_config.get("algorithm", phase_name)),
                base_model=manifest_base_model,
                base_model_source=_default_model_source(phase_name),
                train_data_path=str(phase_train_data_path),
                eval_data_path=(
                    str(phase_eval_data_path) if phase_eval_data_path is not None else None
                ),
                output_dir=str(manifest_output_dir),
                prompt_template=prompt_template,
                reward_profile=reward_profile,
                adapter_config=dict(matrix.finetune),
                precision_config=_extract_precision_config(matrix),
                trainer={
                    key: phase_config[key]
                    for key in sorted(phase_config)
                    if key not in {"algorithm", "reward_profile"}
                },
                runtime=_build_runtime_descriptor(
                    phase_name=phase_name,
                    backend=backend,
                    algorithm=str(phase_config.get("algorithm", phase_name)),
                    common_runtime=common_runtime,
                ),
                notes=notes,
            )
        )
    return manifests


def render_verl_manifest_yaml(manifest: TrainingManifest) -> str:
    payload = {
        "backend": manifest.backend,
        "phase": manifest.phase,
        "algorithm": manifest.algorithm,
        "model": {
            "base_model": manifest.base_model,
            "base_model_source": manifest.base_model_source,
            "adapter_config": manifest.adapter_config,
            "precision_config": manifest.precision_config,
        },
        "data": {
            "train_data_path": manifest.train_data_path,
            "eval_data_path": manifest.eval_data_path,
            "prompt_template": manifest.prompt_template,
        },
        "reward": {
            "profile": manifest.reward_profile,
            "source": "veridoc_rl.verifiers",
        },
        "trainer": manifest.trainer,
        "runtime": manifest.runtime,
        "output": {
            "output_dir": manifest.output_dir,
        },
        "notes": manifest.notes,
    }
    return _render_yaml(payload)


def render_manifest_markdown(manifest: TrainingManifest) -> str:
    trainer_items = [f"- `{key}`: {value}" for key, value in sorted(manifest.trainer.items())]
    note_items = [f"- {item}" for item in manifest.notes]
    runtime_backend = manifest.runtime.get("backend_name")
    return "\n".join(
        [
            f"# {manifest.name}",
            "",
            f"- phase: `{manifest.phase}`",
            f"- backend: `{manifest.backend}`",
            f"- algorithm: `{manifest.algorithm}`",
            f"- base_model: `{manifest.base_model}`",
            f"- base_model_source: `{manifest.base_model_source}`",
            f"- train_data_path: `{manifest.train_data_path}`",
            f"- eval_data_path: `{manifest.eval_data_path}`",
            f"- reward_profile: `{manifest.reward_profile}`",
            "",
            "## Finetune",
            f"- `adapter_type`: `{manifest.adapter_config.get('adapter_type')}`",
            f"- `load_in_4bit`: {manifest.adapter_config.get('load_in_4bit')}",
            f"- `torch_dtype`: `{manifest.precision_config.get('torch_dtype')}`",
            f"- `gradient_checkpointing`: {manifest.precision_config.get('gradient_checkpointing')}",
            "",
            "## Trainer",
            *trainer_items,
            "",
            "## Runtime",
            f"- `runtime_backend`: `{runtime_backend}`",
            f"- `supported`: {manifest.runtime.get('supported', False)}",
            f"- `entrypoint_module`: `{manifest.runtime.get('entrypoint_module')}`",
            f"- `dataset_format`: `{manifest.runtime.get('dataset_format')}`",
            f"- `reward_function`: `{manifest.runtime.get('reward_function')}`",
            "",
            "## Notes",
            *note_items,
            "",
        ]
    )


def write_training_bundle(output_dir: Path, manifests: list[TrainingManifest]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        manifest_dir = output_dir / manifest.name
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (manifest_dir / "README.md").write_text(
            render_manifest_markdown(manifest),
            encoding="utf-8",
        )
        (manifest_dir / "verl.yaml").write_text(
            render_verl_manifest_yaml(manifest),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase A / Phase B / Phase C training manifests."
    )
    parser.add_argument("--matrix-path", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument(
        "--train-data-path",
        type=Path,
        help="Fallback training corpus path applied to every phase.",
    )
    parser.add_argument("--phase-a-train-data-path", type=Path, help="Phase A SFT train corpus.")
    parser.add_argument("--phase-b-train-data-path", type=Path, help="Phase B DPO train corpus.")
    parser.add_argument("--phase-c-train-data-path", type=Path, help="Phase C RLVR train corpus.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the training bundle.",
    )
    parser.add_argument("--eval-data-path", type=Path, help="Optional eval corpus path.")
    parser.add_argument("--phase-a-eval-data-path", type=Path, help="Phase A SFT eval corpus.")
    parser.add_argument("--phase-b-eval-data-path", type=Path, help="Phase B DPO eval corpus.")
    parser.add_argument("--phase-c-eval-data-path", type=Path, help="Phase C RLVR eval corpus.")
    parser.add_argument("--base-model", help="Optional base model override.")
    parser.add_argument("--phase-a-base-model", help="Optional Phase A base model override.")
    parser.add_argument("--phase-b-base-model", help="Optional Phase B base model override.")
    parser.add_argument("--phase-c-base-model", help="Optional Phase C base model override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix = load_experiment_matrix(args.matrix_path)
    manifests = build_training_manifests(
        matrix,
        output_dir=args.output_dir,
        train_data_path=args.train_data_path,
        train_data_paths={
            "phase_a_sft": args.phase_a_train_data_path,
            "phase_b_dpo": args.phase_b_train_data_path,
            "phase_c_grpo": args.phase_c_train_data_path,
            "phase_c_rloo": args.phase_c_train_data_path,
        },
        eval_data_path=args.eval_data_path,
        eval_data_paths={
            "phase_a_sft": args.phase_a_eval_data_path,
            "phase_b_dpo": args.phase_b_eval_data_path,
            "phase_c_grpo": args.phase_c_eval_data_path,
            "phase_c_rloo": args.phase_c_eval_data_path,
        },
        base_model=args.base_model,
        base_models={
            "phase_a_sft": args.phase_a_base_model,
            "phase_b_dpo": args.phase_b_base_model,
            "phase_c_grpo": args.phase_c_base_model,
            "phase_c_rloo": args.phase_c_base_model,
        },
    )
    write_training_bundle(args.output_dir, manifests)
    return 0


def _resolve_runtime_config(matrix: ExperimentMatrix) -> dict[str, Any]:
    runtime = matrix.training_runtime
    if isinstance(runtime, dict):
        merged = dict(DEFAULT_RUNTIME)
        merged.update(runtime)
        return merged
    return dict(DEFAULT_RUNTIME)


def _build_runtime_descriptor(
    *,
    phase_name: str,
    backend: str,
    algorithm: str,
    common_runtime: dict[str, Any],
) -> dict[str, Any]:
    if backend not in {"multi", "trl", "verl"}:
        return {
            "supported": False,
            "backend_name": None,
            "entrypoint_module": None,
            "dataset_format": None,
            "reward_function": None,
            "reason": f"Unsupported runtime backend for this adapter: {backend}",
        }
    if phase_name == "phase_a_sft":
        return {
            "supported": True,
            "backend_name": "transformers",
            "entrypoint_module": "veridoc_rl.training.trl_sft",
            "dataset_format": "jsonl",
            "reward_function": None,
            "launcher_defaults": dict(common_runtime),
            "reason": "",
        }
    if phase_name == "phase_b_dpo":
        return {
            "supported": True,
            "backend_name": "trl",
            "entrypoint_module": "veridoc_rl.training.trl_dpo",
            "dataset_format": "jsonl",
            "reward_function": None,
            "launcher_defaults": dict(common_runtime),
            "reason": "",
        }
    return {
        "supported": True,
        "backend_name": "verl",
        "entrypoint_module": "verl.trainer.main_ppo",
        "dataset_format": "parquet",
        "reward_function": "veridoc_rl.training.verl_reward.compute_score",
        "adv_estimator": algorithm,
        "launcher_defaults": dict(common_runtime),
        "reason": "",
    }


def _render_yaml(value: Any, *, indent: int = 0) -> str:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            prefix = " " * indent + f"{key}:"
            if isinstance(item, (dict, list)):
                lines.append(prefix)
                lines.append(_render_yaml(item, indent=indent + 2))
            else:
                lines.append(prefix + f" {_render_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            prefix = " " * indent + "- "
            if isinstance(item, (dict, list)):
                lines.append(prefix.rstrip())
                lines.append(_render_yaml(item, indent=indent + 2))
            else:
                lines.append(prefix + _render_scalar(item))
        return "\n".join(lines)
    return " " * indent + _render_scalar(value)


def _render_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in [":", "#", "{", "}", "[", "]"]):
        return json.dumps(text, ensure_ascii=False)
    return text


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _resolve_phase_path(
    phase_name: str,
    *,
    legacy_path: Path | None,
    phase_paths: Mapping[str, Path] | None,
) -> Path | None:
    if phase_paths is not None:
        phase_value = phase_paths.get(phase_name)
        if phase_value is not None:
            return phase_value
    return legacy_path


def _resolve_phase_base_model(
    phase_name: str,
    *,
    matrix: ExperimentMatrix,
    legacy_model: str | None,
    phase_models: Mapping[str, str] | None,
) -> str:
    if phase_models is not None and phase_models.get(phase_name):
        return str(phase_models[phase_name])
    if legacy_model is not None:
        return legacy_model
    if phase_name == "phase_a_sft":
        return matrix.base_model.get("mvp") or DEFAULT_BASELINE_MODEL
    return matrix.base_model.get("full") or matrix.base_model.get("mvp") or DEFAULT_BASELINE_MODEL


def _extract_precision_config(matrix: ExperimentMatrix) -> dict[str, Any]:
    precision = matrix.finetune.get("precision")
    if isinstance(precision, dict):
        return dict(precision)
    return {}


def _default_model_source(phase_name: str) -> str:
    if phase_name == "phase_a_sft":
        return "baseline"
    return "sft_checkpoint"


if __name__ == "__main__":
    raise SystemExit(main())
