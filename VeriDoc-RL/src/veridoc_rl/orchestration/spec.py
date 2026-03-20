from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.experiments.matrix import _parse_simple_yaml
from veridoc_rl.path_utils import expand_env_and_user
from veridoc_rl.training.prompting import DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True, frozen=True)
class RunConfig:
    name: str
    output_root: str = "outputs/pipelines"


@dataclass(slots=True, frozen=True)
class ModelConfig:
    baseline: str
    inference_backend: str = "sglang"
    inference_api_base: str = "http://127.0.0.1:30000/v1"
    inference_api_key: str = "EMPTY"


@dataclass(slots=True, frozen=True)
class DataConfig:
    sft_gold_path: str
    rl_prompt_only_path: str


@dataclass(slots=True, frozen=True)
class GenerationConfig:
    candidate_count: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    enable_baseline_eval: bool = True
    enable_sft: bool = True
    enable_dpo: bool = True
    enable_rl: bool = True
    rl_algorithm: str = "grpo"


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    prepare_only: bool = False
    execute_training: bool = True
    resume: bool = True


@dataclass(slots=True, frozen=True)
class PipelineSpec:
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    generation: GenerationConfig
    pipeline: PipelineConfig
    execution: ExecutionConfig
    matrix_path: str = "configs/experiment_matrix.yaml"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pipeline_spec(path: Path) -> PipelineSpec:
    text = path.read_text(encoding="utf-8-sig")
    payload = json.loads(text) if path.suffix == ".json" else _parse_simple_yaml(text)
    if not isinstance(payload, dict):
        raise ValueError("Pipeline spec root must be a mapping.")

    run_payload = _require_mapping(payload.get("run"), "run")
    model_payload = _require_mapping(payload.get("model"), "model")
    data_payload = _require_mapping(payload.get("data"), "data")
    generation_payload = _optional_mapping(payload.get("generation")) or {}
    pipeline_payload = _optional_mapping(payload.get("pipeline")) or {}
    execution_payload = _optional_mapping(payload.get("execution")) or {}

    return PipelineSpec(
        run=RunConfig(
            name=expand_env_and_user(str(run_payload["name"])),
            output_root=expand_env_and_user(str(run_payload.get("output_root", "outputs/pipelines"))),
        ),
        model=ModelConfig(
            baseline=expand_env_and_user(str(model_payload["baseline"])),
            inference_backend=expand_env_and_user(
                str(model_payload.get("inference_backend", "sglang"))
            ),
            inference_api_base=expand_env_and_user(str(
                model_payload.get("inference_api_base", "http://127.0.0.1:30000/v1")
            )),
            inference_api_key=expand_env_and_user(
                str(model_payload.get("inference_api_key", "EMPTY"))
            ),
        ),
        data=DataConfig(
            sft_gold_path=expand_env_and_user(str(data_payload["sft_gold_path"])),
            rl_prompt_only_path=expand_env_and_user(str(data_payload["rl_prompt_only_path"])),
        ),
        generation=GenerationConfig(
            candidate_count=int(generation_payload.get("candidate_count", 4)),
            temperature=float(generation_payload.get("temperature", 0.8)),
            top_p=float(generation_payload.get("top_p", 0.95)),
            max_new_tokens=int(generation_payload.get("max_new_tokens", 1024)),
            system_prompt=str(generation_payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        ),
        pipeline=PipelineConfig(
            enable_baseline_eval=bool(pipeline_payload.get("enable_baseline_eval", True)),
            enable_sft=bool(pipeline_payload.get("enable_sft", True)),
            enable_dpo=bool(pipeline_payload.get("enable_dpo", True)),
            enable_rl=bool(pipeline_payload.get("enable_rl", True)),
            rl_algorithm=str(pipeline_payload.get("rl_algorithm", "grpo")),
        ),
        execution=ExecutionConfig(
            prepare_only=bool(execution_payload.get("prepare_only", False)),
            execute_training=bool(execution_payload.get("execute_training", True)),
            resume=bool(execution_payload.get("resume", True)),
        ),
        matrix_path=expand_env_and_user(str(payload.get("matrix_path", "configs/experiment_matrix.yaml"))),
    )


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
