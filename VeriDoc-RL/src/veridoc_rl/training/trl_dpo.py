from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import load_jsonl


@dataclass(slots=True, frozen=True)
class TrlDPOConfig:
    model_name_or_path: str
    train_data_path: str
    eval_data_path: str | None
    output_dir: str
    learning_rate: float
    beta: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    max_prompt_length: int
    max_completion_length: int
    logging_steps: int
    save_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_trl_dpo_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = str(record.get("prompt", "")).strip()
        system_prompt = str(record.get("system_prompt", "")).strip()
        chosen = str(record.get("chosen", "")).strip()
        rejected = str(record.get("rejected", "")).strip()
        sample_id = str(record.get("sample_id", f"sample-{index}"))
        if not prompt:
            raise ValueError(f"DPO record {sample_id} is missing prompt text.")
        if not chosen:
            raise ValueError(f"DPO record {sample_id} is missing chosen response text.")
        if not rejected:
            raise ValueError(f"DPO record {sample_id} is missing rejected response text.")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        rows.append(
            {
                "prompt": full_prompt.strip(),
                "chosen": chosen,
                "rejected": rejected,
                "sample_id": sample_id,
                "reward_profile": str(record.get("reward_profile", "default")),
                "reward_margin": float(record.get("reward_margin", 0.0)),
                "chosen_candidate_id": str(record.get("chosen_candidate_id", "")),
                "rejected_candidate_id": str(record.get("rejected_candidate_id", "")),
                "metadata": json.dumps(record.get("metadata", {}), ensure_ascii=False),
            }
        )
    return rows


def export_trl_dpo_dataset(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = build_trl_dpo_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_trl_dpo_config(path: Path, config: TrlDPOConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_trl_dpo_config(path: Path) -> TrlDPOConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TRL DPO config at {path} must be a JSON object.")
    return TrlDPOConfig(
        model_name_or_path=str(payload["model_name_or_path"]),
        train_data_path=str(payload["train_data_path"]),
        eval_data_path=(
            str(payload["eval_data_path"]) if payload.get("eval_data_path") is not None else None
        ),
        output_dir=str(payload["output_dir"]),
        learning_rate=float(payload["learning_rate"]),
        beta=float(payload["beta"]),
        num_train_epochs=float(payload["num_train_epochs"]),
        per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
        max_length=int(payload["max_length"]),
        max_prompt_length=int(payload["max_prompt_length"]),
        max_completion_length=int(payload["max_completion_length"]),
        logging_steps=int(payload["logging_steps"]),
        save_steps=int(payload["save_steps"]),
    )


def execute_trl_dpo_training(config: TrlDPOConfig) -> int:
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Executing the DPO runtime requires the optional training dependencies "
            "(`trl`, `transformers`, `datasets`, `accelerate`, and usually `peft`)."
        ) from exc

    train_dataset = Dataset.from_list(build_trl_dpo_rows(load_jsonl(Path(config.train_data_path))))
    eval_dataset = None
    if config.eval_data_path is not None:
        eval_dataset = Dataset.from_list(
            build_trl_dpo_rows(load_jsonl(Path(config.eval_data_path)))
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    training_args = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=[],
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    init_signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in init_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in init_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("Unsupported TRL version: DPOTrainer does not accept tokenizer input.")

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(config.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a TRL-backed DPO training run from a config file."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the generated DPO config JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return execute_trl_dpo_training(load_trl_dpo_config(args.config_path))


if __name__ == "__main__":
    raise SystemExit(main())
