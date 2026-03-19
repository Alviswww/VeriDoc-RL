from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class StageState:
    status: str = "pending"
    base_model: str | None = None
    train_data_path: str | None = None
    manifest_path: str | None = None
    runtime_plan_path: str | None = None
    checkpoint_path: str | None = None
    candidate_path: str | None = None
    prediction_path: str | None = None
    report_path: str | None = None
    case_export_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PipelineState:
    run_name: str
    status: str = "pending"
    stages: dict[str, StageState] = field(default_factory=dict)

    def ensure_stage(self, stage_name: str) -> StageState:
        stage = self.stages.get(stage_name)
        if stage is None:
            stage = StageState()
            self.stages[stage_name] = stage
        return stage

    def to_dict(self) -> dict[str, object]:
        return {
            "run_name": self.run_name,
            "status": self.status,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
        }


def load_or_create_state(path: Path, *, run_name: str, stage_names: list[str]) -> PipelineState:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = PipelineState(
            run_name=str(payload.get("run_name", run_name)),
            status=str(payload.get("status", "pending")),
            stages={
                name: StageState(**dict(value))
                for name, value in dict(payload.get("stages", {})).items()
                if isinstance(value, dict)
            },
        )
    else:
        state = PipelineState(run_name=run_name)
    for stage_name in stage_names:
        state.ensure_stage(stage_name)
    return state


def save_state(path: Path, state: PipelineState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def mark_stage_running(stage: StageState, *, base_model: str | None = None) -> None:
    stage.status = "running"
    stage.error = ""
    stage.started_at = _timestamp()
    if base_model is not None:
        stage.base_model = base_model


def mark_stage_succeeded(stage: StageState) -> None:
    stage.status = "succeeded"
    stage.finished_at = _timestamp()
    stage.error = ""


def mark_stage_failed(stage: StageState, message: str) -> None:
    stage.status = "failed"
    stage.finished_at = _timestamp()
    stage.error = message


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
