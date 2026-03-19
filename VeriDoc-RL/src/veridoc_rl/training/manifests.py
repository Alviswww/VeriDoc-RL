from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.experiments import ExperimentMatrix, load_experiment_matrix

DEFAULT_RUNTIME = {
    "backend": "multi",
    "prompt_template": "veridoc_v1",
    "phases": {
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
    train_data_path: str
    eval_data_path: str | None
    output_dir: str
    prompt_template: str
    reward_profile: str
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
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            "output_dir": self.output_dir,
            "prompt_template": self.prompt_template,
            "reward_profile": self.reward_profile,
            "trainer": dict(self.trainer),
            "runtime": dict(self.runtime),
            "notes": list(self.notes),
        }


def build_training_manifests(
    matrix: ExperimentMatrix,
    *,
    train_data_path: Path,
    output_dir: Path,
    eval_data_path: Path | None = None,
    base_model: str | None = None,
) -> list[TrainingManifest]:
    runtime = _resolve_runtime_config(matrix)
    backend = str(runtime.get("backend", "verl"))
    prompt_template = str(runtime.get("prompt_template", "veridoc_v1"))
    phase_configs = _as_mapping(runtime.get("phases"), field_name="training_runtime.phases")
    common_runtime = _as_mapping(runtime.get("common", {}), field_name="training_runtime.common")

    manifests: list[TrainingManifest] = []
    for phase_name, config in phase_configs.items():
        phase_config = _as_mapping(config, field_name=f"training_runtime.phases.{phase_name}")
        manifest_base_model = (
            base_model
            or matrix.base_model.get("full")
            or matrix.base_model.get("mvp")
            or "Qwen2.5-7B-Instruct"
        )
        reward_profile = str(phase_config.get("reward_profile", "default"))
        manifest_output_dir = output_dir / phase_name
        notes = [
            (
                "The same prompt template and verifier reward profile should be reused "
                "across train/eval/reporting."
            ),
        ]
        manifests.append(
            TrainingManifest(
                name=phase_name,
                phase=phase_name,
                backend=backend,
                algorithm=str(phase_config.get("algorithm", phase_name)),
                base_model=manifest_base_model,
                train_data_path=str(train_data_path),
                eval_data_path=str(eval_data_path) if eval_data_path is not None else None,
                output_dir=str(manifest_output_dir),
                prompt_template=prompt_template,
                reward_profile=reward_profile,
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
            f"- train_data_path: `{manifest.train_data_path}`",
            f"- eval_data_path: `{manifest.eval_data_path}`",
            f"- reward_profile: `{manifest.reward_profile}`",
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
        description="Generate Phase B/Phase C training manifests."
    )
    parser.add_argument("--matrix-path", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument(
        "--train-data-path",
        type=Path,
        required=True,
        help="Prepared DPO or RLVR training corpus path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the training bundle.",
    )
    parser.add_argument("--eval-data-path", type=Path, help="Optional eval corpus path.")
    parser.add_argument("--base-model", help="Optional base model override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix = load_experiment_matrix(args.matrix_path)
    manifests = build_training_manifests(
        matrix,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
        base_model=args.base_model,
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


if __name__ == "__main__":
    raise SystemExit(main())
