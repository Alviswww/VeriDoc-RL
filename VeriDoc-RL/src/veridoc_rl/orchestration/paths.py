from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from veridoc_rl.orchestration.spec import PipelineSpec


@dataclass(slots=True, frozen=True)
class PipelinePaths:
    root: Path

    @classmethod
    def from_spec(cls, spec: PipelineSpec) -> "PipelinePaths":
        return cls(root=Path(spec.run.output_root) / spec.run.name)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def spec_snapshot_path(self) -> Path:
        return self.root / "spec.snapshot.json"

    @property
    def comparison_dir(self) -> Path:
        return self.root / "comparison"

    def stage_dir(self, stage_name: str) -> Path:
        return self.root / stage_name

    def stage_train_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "train.jsonl"

    def stage_candidates_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "candidates.jsonl"

    def stage_preferences_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "preferences.jsonl"

    def stage_predictions_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "predictions.jsonl"

    def stage_report_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "report.json"

    def stage_cases_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "cases.jsonl"

    def stage_manifest_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "manifest.json"

    def stage_runtime_plan_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "runtime_plan.json"

    def stage_launch_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "launch.sh"

    def stage_checkpoint_dir(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "checkpoints"
