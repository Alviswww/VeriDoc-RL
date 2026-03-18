from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MATRIX_PATH = Path("configs/experiment_matrix.yaml")
_ABLATION_STAGE_MAP: dict[str, str] = {
    "sft_only": "sft_baseline",
    "sft_plus_dpo": "preference_optimization",
    "sft_plus_rlvr": "rlvr",
    "sft_plus_rlvr_without_cross_field_consistency": "rlvr",
    "sft_plus_rlvr_without_checkbox_logic": "rlvr",
}
_ABLATION_REWARD_PROFILE_MAP: dict[str, str] = {
    "sft_only": "default",
    "sft_plus_dpo": "default",
    "sft_plus_rlvr": "rlvr",
    "sft_plus_rlvr_without_cross_field_consistency": "rlvr_without_cross_field_consistency",
    "sft_plus_rlvr_without_checkbox_logic": "rlvr_without_checkbox_logic",
}
_DEFAULT_MODEL_TIER_BY_STAGE: dict[str, str] = {
    "sft_baseline": "mvp",
    "preference_optimization": "mvp",
    "rlvr": "full",
}


@dataclass(slots=True, frozen=True)
class TrainingStage:
    name: str
    method: str
    goal: str


@dataclass(slots=True, frozen=True)
class ExperimentMatrix:
    project: dict[str, Any]
    base_model: dict[str, str]
    data: dict[str, Any]
    training_stages: tuple[TrainingStage, ...]
    reward: dict[str, Any]
    evaluation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": dict(self.project),
            "base_model": dict(self.base_model),
            "data": dict(self.data),
            "training_stages": [
                {"name": stage.name, "method": stage.method, "goal": stage.goal}
                for stage in self.training_stages
            ],
            "reward": dict(self.reward),
            "evaluation": dict(self.evaluation),
        }


def load_experiment_matrix(path: Path = DEFAULT_MATRIX_PATH) -> ExperimentMatrix:
    payload = _parse_simple_yaml(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment matrix root must be a mapping.")

    training_stages_payload = payload.get("training_stages", [])
    if not isinstance(training_stages_payload, list):
        raise ValueError("training_stages must be a list.")

    training_stages = tuple(
        TrainingStage(
            name=str(stage["name"]),
            method=str(stage["method"]),
            goal=str(stage["goal"]),
        )
        for stage in training_stages_payload
        if isinstance(stage, dict)
    )
    if not training_stages:
        raise ValueError("training_stages must contain at least one stage definition.")

    return ExperimentMatrix(
        project=_as_mapping(payload.get("project"), field_name="project"),
        base_model={key: str(value) for key, value in _as_mapping(payload.get("base_model"), field_name="base_model").items()},
        data=_as_mapping(payload.get("data"), field_name="data"),
        training_stages=training_stages,
        reward=_as_mapping(payload.get("reward"), field_name="reward"),
        evaluation=_as_mapping(payload.get("evaluation"), field_name="evaluation"),
    )


def build_experiment_plan(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    stages_by_name = {stage.name: stage for stage in matrix.training_stages}
    ablations = matrix.evaluation.get("ablations", [])
    primary_metrics = list(_as_list(matrix.evaluation.get("primary_metrics"), field_name="evaluation.primary_metrics"))
    buckets = _as_mapping(matrix.data.get("buckets"), field_name="data.buckets")
    task_scope = list(_as_list(matrix.data.get("task_scope"), field_name="data.task_scope"))

    plan: list[dict[str, Any]] = []
    for ablation_name in _as_list(ablations, field_name="evaluation.ablations"):
        ablation_key = str(ablation_name)
        stage_name = _ABLATION_STAGE_MAP.get(ablation_key)
        if stage_name is None or stage_name not in stages_by_name:
            raise ValueError(f"Unsupported ablation stage mapping: {ablation_key}")
        stage = stages_by_name[stage_name]
        model_tier = _DEFAULT_MODEL_TIER_BY_STAGE.get(stage.name, "mvp")
        plan.append(
            {
                "experiment_name": ablation_key,
                "training_stage": stage.name,
                "training_method": stage.method,
                "goal": stage.goal,
                "recommended_model_tier": model_tier,
                "recommended_model": matrix.base_model.get(model_tier),
                "reward_profile": _ABLATION_REWARD_PROFILE_MAP.get(ablation_key, "default"),
                "task_scope": task_scope,
                "bucket_dimensions": {
                    key: list(_as_list(value, field_name=f"data.buckets.{key}"))
                    for key, value in buckets.items()
                },
                "primary_metrics": primary_metrics,
            }
        )
    return plan


def render_experiment_plan_markdown(
    matrix: ExperimentMatrix,
    experiment_plan: list[dict[str, Any]],
) -> str:
    bucket_lines = [
        f"- `{dimension}`: {', '.join(str(item) for item in _as_list(values, field_name=dimension))}"
        for dimension, values in _as_mapping(matrix.data.get("buckets"), field_name="data.buckets").items()
    ]
    reward_components = _as_mapping(matrix.reward.get("components"), field_name="reward.components")
    reward_lines = [
        f"- `{name}`: {float(weight):.2f}" for name, weight in reward_components.items()
    ]
    stage_lines = [
        f"- `{stage.name}` / `{stage.method}`: {stage.goal}" for stage in matrix.training_stages
    ]
    experiment_lines = [
        (
            f"- `{item['experiment_name']}`: stage=`{item['training_stage']}`, "
            f"reward_profile=`{item['reward_profile']}`, model=`{item['recommended_model']}`"
        )
        for item in experiment_plan
    ]
    metrics = ", ".join(
        f"`{metric}`"
        for metric in _as_list(matrix.evaluation.get("primary_metrics"), field_name="evaluation.primary_metrics")
    )
    return "\n".join(
        [
            f"# {matrix.project.get('name', 'VeriDoc-RL')} Experiment Plan",
            "",
            f"- Objective: {matrix.project.get('objective', '')}",
            f"- Primary metrics: {metrics}",
            "",
            "## Training Stages",
            *stage_lines,
            "",
            "## Bucket Design",
            *bucket_lines,
            "",
            "## Reward Components",
            *reward_lines,
            "",
            "## Recommended Experiment Queue",
            *experiment_lines,
            "",
        ]
    )


def write_experiment_plan(
    output_path: Path,
    matrix: ExperimentMatrix,
    experiment_plan: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix": matrix.to_dict(),
        "experiments": experiment_plan,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the configured experiment matrix into an executable plan.")
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH, help="Path to configs/experiment_matrix.yaml.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON path for the expanded experiment plan.")
    parser.add_argument("--markdown-path", type=Path, help="Optional markdown summary path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix = load_experiment_matrix(args.matrix_path)
    plan = build_experiment_plan(matrix)
    write_experiment_plan(args.output_path, matrix, plan)
    if args.markdown_path is not None:
        args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_path.write_text(
            render_experiment_plan_markdown(matrix, plan),
            encoding="utf-8",
        )
    return 0


def _parse_simple_yaml(text: str) -> Any:
    lines = _tokenize_yaml_lines(text)
    if not lines:
        return {}
    value, next_index = _parse_block(lines, start=0, indent=0)
    if next_index != len(lines):
        raise ValueError("Unexpected trailing YAML content.")
    return value


def _tokenize_yaml_lines(text: str) -> list[tuple[int, str]]:
    tokenized: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip(" ")
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(stripped)
        tokenized.append((indent, stripped))
    return tokenized


def _parse_block(
    lines: list[tuple[int, str]],
    *,
    start: int,
    indent: int,
) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start
    current_indent, current_content = lines[start]
    if current_indent != indent:
        raise ValueError(f"Invalid indentation at line fragment: {current_content}")
    if current_content.startswith("- "):
        return _parse_list(lines, start=start, indent=indent)
    return _parse_mapping(lines, start=start, indent=indent)


def _parse_list(
    lines: list[tuple[int, str]],
    *,
    start: int,
    indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            raise ValueError(f"Invalid list indentation near: {content}")
        if not content.startswith("- "):
            break

        item_content = content[2:].strip()
        if not item_content:
            item, index = _parse_block(lines, start=index + 1, indent=indent + 2)
            items.append(item)
            continue

        if ":" in item_content:
            key, rest = _split_mapping_entry(item_content)
            item: dict[str, Any] = {}
            if rest is None:
                nested_value, index = _parse_block(lines, start=index + 1, indent=indent + 2)
                item[key] = nested_value
            else:
                item[key] = _parse_scalar(rest)
                index += 1

            while index < len(lines):
                next_indent, next_content = lines[index]
                if next_indent < indent + 2:
                    break
                if next_indent == indent and next_content.startswith("- "):
                    break
                if next_indent != indent + 2:
                    raise ValueError(f"Invalid nested mapping indentation near: {next_content}")
                nested_key, nested_rest = _split_mapping_entry(next_content)
                if nested_rest is None:
                    nested_value, index = _parse_block(lines, start=index + 1, indent=indent + 4)
                    item[nested_key] = nested_value
                    continue
                item[nested_key] = _parse_scalar(nested_rest)
                index += 1
            items.append(item)
            continue

        items.append(_parse_scalar(item_content))
        index += 1
    return items, index


def _parse_mapping(
    lines: list[tuple[int, str]],
    *,
    start: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            raise ValueError(f"Invalid mapping indentation near: {content}")
        if content.startswith("- "):
            break

        key, rest = _split_mapping_entry(content)
        if rest is None:
            if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                mapping[key] = {}
                index += 1
                continue
            value, index = _parse_block(lines, start=index + 1, indent=indent + 2)
            mapping[key] = value
            continue
        mapping[key] = _parse_scalar(rest)
        index += 1
    return mapping, index


def _split_mapping_entry(content: str) -> tuple[str, str | None]:
    key, separator, remainder = content.partition(":")
    if not separator:
        raise ValueError(f"Invalid mapping entry: {content}")
    stripped_key = key.strip()
    stripped_rest = remainder.strip()
    return stripped_key, stripped_rest or None


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    if value.replace(".", "", 1).isdigit() and value.count(".") < 2:
        if "." in value:
            return float(value)
        return int(value)
    return value


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
