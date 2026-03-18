from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import load_jsonl
from veridoc_rl.training.manifests import TrainingManifest


DEFAULT_VERL_RUNTIME = {
    "project_name": "VeriDoc-RL",
    "python_bin": "python3",
    "n_gpus_per_node": 1,
    "nnodes": 1,
    "prompt_key": "prompt",
    "max_prompt_length": 2048,
    "max_response_length": 1024,
    "train_batch_size": 32,
    "val_batch_size": 32,
    "filter_overlong_prompts": False,
    "truncation": "left",
    "rollout_engine": "vllm",
    "rollout_mode": "sync",
    "rollout_tensor_parallel_size": 1,
    "rollout_gpu_memory_utilization": 0.5,
    "ppo_mini_batch_size": 16,
    "ppo_micro_batch_size_per_gpu": 4,
    "log_prob_micro_batch_size_per_gpu": 4,
    "save_freq": 10,
    "test_freq": 10,
    "reward_manager": "naive",
}


@dataclass(slots=True, frozen=True)
class RuntimeLaunchPlan:
    supported: bool
    manifest_name: str
    phase: str
    run_dir: str
    command: list[str]
    command_preview: str
    train_data_path: str | None
    eval_data_path: str | None
    generated_files: list[str]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "manifest_name": self.manifest_name,
            "phase": self.phase,
            "run_dir": self.run_dir,
            "command": list(self.command),
            "command_preview": self.command_preview,
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            "generated_files": list(self.generated_files),
            "reason": self.reason,
        }


def load_training_manifest(path: Path) -> TrainingManifest:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest at {path} must be a JSON object.")
    return TrainingManifest(
        name=str(payload["name"]),
        phase=str(payload["phase"]),
        backend=str(payload["backend"]),
        algorithm=str(payload["algorithm"]),
        base_model=str(payload["base_model"]),
        train_data_path=str(payload["train_data_path"]),
        eval_data_path=(
            str(payload["eval_data_path"]) if payload.get("eval_data_path") is not None else None
        ),
        output_dir=str(payload["output_dir"]),
        prompt_template=str(payload["prompt_template"]),
        reward_profile=str(payload["reward_profile"]),
        trainer=dict(payload.get("trainer", {})),
        runtime=dict(payload.get("runtime", {})),
        notes=[str(item) for item in payload.get("notes", [])],
    )


def build_runtime_launch_plan(
    manifest: TrainingManifest,
    *,
    run_dir: Path,
    materialize_data: bool = False,
    project_name: str | None = None,
    experiment_name: str | None = None,
    n_gpus_per_node: int | None = None,
    nnodes: int | None = None,
    python_bin: str | None = None,
    extra_overrides: list[str] | None = None,
) -> RuntimeLaunchPlan:
    runtime_info = dict(DEFAULT_VERL_RUNTIME)
    runtime_info.update(dict(manifest.runtime.get("launcher_defaults", {})))
    runtime_info.update(
        {
            key: value
            for key, value in manifest.trainer.items()
            if key in runtime_info
        }
    )
    if project_name is not None:
        runtime_info["project_name"] = project_name
    if experiment_name is not None:
        runtime_info["experiment_name"] = experiment_name
    else:
        runtime_info["experiment_name"] = manifest.name
    if n_gpus_per_node is not None:
        runtime_info["n_gpus_per_node"] = n_gpus_per_node
    if nnodes is not None:
        runtime_info["nnodes"] = nnodes
    if python_bin is not None:
        runtime_info["python_bin"] = python_bin

    run_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    if not bool(manifest.runtime.get("supported")):
        return RuntimeLaunchPlan(
            supported=False,
            manifest_name=manifest.name,
            phase=manifest.phase,
            run_dir=str(run_dir),
            command=[],
            command_preview="",
            train_data_path=None,
            eval_data_path=None,
            generated_files=generated_files,
            reason=str(manifest.runtime.get("reason", "Unsupported runtime manifest.")),
        )

    staged_train = _stage_dataset(
        source=Path(manifest.train_data_path),
        run_dir=run_dir,
        stage=manifest.phase,
        split="train",
        materialize_data=materialize_data,
    )
    notes: list[str] = []
    if Path(manifest.train_data_path).suffix == ".jsonl" and not materialize_data:
        notes.append(
            "Train corpus is JSONL. Materialize parquet before execution by rerunning with --materialize-data."
        )
    staged_eval = None
    if manifest.eval_data_path is not None:
        staged_eval = _stage_dataset(
            source=Path(manifest.eval_data_path),
            run_dir=run_dir,
            stage=manifest.phase,
            split="eval",
            materialize_data=materialize_data,
        )
        if Path(manifest.eval_data_path).suffix == ".jsonl" and not materialize_data:
            notes.append(
                "Eval corpus is JSONL. Materialize parquet before execution by rerunning with --materialize-data."
            )

    reward_file = Path(__file__).with_name("verl_reward.py")
    generated_files.append(str(reward_file))
    command = [
        str(runtime_info["python_bin"]),
        "-m",
        str(manifest.runtime["entrypoint_module"]),
        f"algorithm.adv_estimator={manifest.runtime['adv_estimator']}",
        f"data.train_files={staged_train}",
        f"data.prompt_key={runtime_info['prompt_key']}",
        f"data.train_batch_size={runtime_info['train_batch_size']}",
        f"data.max_prompt_length={runtime_info['max_prompt_length']}",
        f"data.max_response_length={runtime_info['max_response_length']}",
        f"data.filter_overlong_prompts={runtime_info['filter_overlong_prompts']}",
        f"data.truncation={runtime_info['truncation']}",
        f"actor_rollout_ref.model.path={manifest.base_model}",
        f"actor_rollout_ref.actor.optim.lr={manifest.trainer.get('learning_rate', 1e-6)}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={runtime_info['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={runtime_info['ppo_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={runtime_info['log_prob_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.rollout.name={runtime_info['rollout_engine']}",
        f"actor_rollout_ref.rollout.mode={runtime_info['rollout_mode']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={runtime_info['rollout_tensor_parallel_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={runtime_info['rollout_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.n={manifest.trainer.get('rollout_n', 4)}",
        f"actor_rollout_ref.rollout.val_kwargs.n=1",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={runtime_info['log_prob_micro_batch_size_per_gpu']}",
        "algorithm.use_kl_in_reward=False",
        f"algorithm.kl_ctrl.kl_coef={manifest.trainer.get('kl_coef', 0.0)}",
        f"reward_model.reward_manager={runtime_info['reward_manager']}",
        f"custom_reward_function.path={reward_file}",
        "custom_reward_function.name=compute_score",
        "trainer.val_before_train=False",
        f"trainer.project_name={runtime_info['project_name']}",
        f"trainer.experiment_name={runtime_info['experiment_name']}",
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.n_gpus_per_node={runtime_info['n_gpus_per_node']}",
        f"trainer.nnodes={runtime_info['nnodes']}",
        f"trainer.save_freq={runtime_info['save_freq']}",
        f"trainer.test_freq={runtime_info['test_freq']}",
        f"trainer.total_epochs={manifest.trainer.get('epochs', 1)}",
    ]
    if staged_eval is not None:
        command.insert(4, f"data.val_files={staged_eval}")
    if extra_overrides:
        command.extend(extra_overrides)
    generated_files.extend([str(run_dir / "runtime_plan.json"), str(run_dir / "launch.sh")])
    return RuntimeLaunchPlan(
        supported=True,
        manifest_name=manifest.name,
        phase=manifest.phase,
        run_dir=str(run_dir),
        command=command,
        command_preview=" ".join(shlex.quote(part) for part in command),
        train_data_path=staged_train,
        eval_data_path=staged_eval,
        generated_files=generated_files,
        reason=" ".join(notes),
    )


def write_runtime_bundle(run_dir: Path, plan: RuntimeLaunchPlan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runtime_plan.json").write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    launch_script = _build_launch_script(plan)
    (run_dir / "launch.sh").write_text(launch_script, encoding="utf-8")


def execute_runtime_plan(plan: RuntimeLaunchPlan) -> int:
    if not plan.supported:
        raise RuntimeError(plan.reason or "Unsupported runtime plan.")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("verl is not installed in the active Python environment.")
    return subprocess.run(plan.command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or execute a verl runtime launch plan from a training manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Path to manifest.json.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for launch artifacts and staged data.")
    parser.add_argument("--materialize-data", action="store_true", help="Convert JSONL training corpora into parquet when needed.")
    parser.add_argument("--execute", action="store_true", help="Execute the generated runtime command after preparing artifacts.")
    parser.add_argument("--project-name", help="Optional trainer.project_name override.")
    parser.add_argument("--experiment-name", help="Optional trainer.experiment_name override.")
    parser.add_argument("--n-gpus-per-node", type=int, help="Override trainer.n_gpus_per_node.")
    parser.add_argument("--nnodes", type=int, help="Override trainer.nnodes.")
    parser.add_argument("--python-bin", help="Override python executable used for the launch command.")
    parser.add_argument("--override", action="append", default=[], help="Extra hydra override appended to the launch command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = load_training_manifest(args.manifest_path)
    plan = build_runtime_launch_plan(
        manifest,
        run_dir=args.run_dir,
        materialize_data=args.materialize_data or args.execute,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        n_gpus_per_node=args.n_gpus_per_node,
        nnodes=args.nnodes,
        python_bin=args.python_bin,
        extra_overrides=list(args.override),
    )
    write_runtime_bundle(args.run_dir, plan)
    if args.execute:
        return execute_runtime_plan(plan)
    return 0 if plan.supported else 1


def _stage_dataset(
    *,
    source: Path,
    run_dir: Path,
    stage: str,
    split: str,
    materialize_data: bool,
) -> str:
    if source.suffix == ".parquet":
        return str(source)
    target = run_dir / "data" / f"{stage}.{split}.parquet"
    if materialize_data:
        materialize_verl_dataset(source=source, target=target, stage=stage)
    return str(target)


def materialize_verl_dataset(*, source: Path, target: Path, stage: str) -> None:
    if source.suffix == ".parquet":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return
    if source.suffix != ".jsonl":
        raise ValueError(f"Unsupported source dataset format for verl staging: {source}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Materializing a JSONL training corpus for verl requires pyarrow. "
            "Install pyarrow in the training environment or pass a parquet dataset directly."
        ) from exc

    rows = _build_verl_rows(load_jsonl(source), stage=stage)
    table = pa.Table.from_pylist(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


def _build_verl_rows(records: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    if stage not in {"phase_c_grpo", "phase_c_rloo"}:
        raise ValueError(f"verl dataset staging is only implemented for RL phases, got: {stage}")
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = str(record.get("prompt", ""))
        system_prompt = str(record.get("system_prompt", ""))
        full_prompt = system_prompt.strip() + "\n\n" + prompt.strip() if system_prompt else prompt
        reference = record.get("reference")
        rows.append(
            {
                "data_source": "veridoc_rl",
                "prompt": full_prompt.strip(),
                "ground_truth": json.dumps(reference, ensure_ascii=False)
                if reference is not None
                else "{}",
                "extra_info": json.dumps(
                    {
                        "sample_id": record.get("sample_id"),
                        "reward_profile": record.get("reward_profile", "rlvr"),
                        "metadata": record.get("metadata", {}),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _build_launch_script(plan: RuntimeLaunchPlan) -> str:
    if not plan.supported:
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"echo {shlex.quote(plan.reason or 'Unsupported runtime plan.')}",
                "",
            ]
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            plan.command_preview,
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
