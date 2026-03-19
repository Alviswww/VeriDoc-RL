from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import load_jsonl
from veridoc_rl.training.finetune import (
    AdapterConfig,
    PrecisionConfig,
    adapter_config_from_mapping,
    load_causal_lm,
    load_tokenizer,
    precision_config_from_mapping,
    render_chat_messages,
    resolve_torch_dtype,
)


@dataclass(slots=True, frozen=True)
class TrlSFTConfig:
    model_name_or_path: str
    train_data_path: str
    eval_data_path: str | None
    output_dir: str
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    logging_steps: int
    save_steps: int
    adapter_config: dict[str, Any]
    precision_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sft_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"SFT record {index} is missing chat `messages`.")
        rows.append(
            {
                "sample_id": str(record.get("sample_id", f"sample-{index}")),
                "messages": list(messages),
                "metadata": dict(record.get("metadata", {}))
                if isinstance(record.get("metadata"), Mapping)
                else {},
            }
        )
    return rows


def export_sft_dataset(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = build_sft_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_trl_sft_config(path: Path, config: TrlSFTConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_trl_sft_config(path: Path) -> TrlSFTConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SFT config at {path} must be a JSON object.")
    return TrlSFTConfig(
        model_name_or_path=str(payload["model_name_or_path"]),
        train_data_path=str(payload["train_data_path"]),
        eval_data_path=(
            str(payload["eval_data_path"]) if payload.get("eval_data_path") is not None else None
        ),
        output_dir=str(payload["output_dir"]),
        learning_rate=float(payload["learning_rate"]),
        num_train_epochs=float(payload["num_train_epochs"]),
        per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
        max_length=int(payload["max_length"]),
        logging_steps=int(payload["logging_steps"]),
        save_steps=int(payload["save_steps"]),
        adapter_config=dict(payload.get("adapter_config", {})),
        precision_config=dict(payload.get("precision_config", {})),
    )


def execute_trl_sft_training(config: TrlSFTConfig) -> int:
    try:
        from datasets import Dataset
        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Executing the SFT runtime requires the optional training dependencies "
            "(`transformers`, `datasets`, `accelerate`, and usually `peft`)."
        ) from exc

    adapter_config = adapter_config_from_mapping(config.adapter_config)
    precision_config = precision_config_from_mapping(config.precision_config)
    tokenizer = load_tokenizer(config.model_name_or_path)
    train_rows = _render_rows(build_sft_rows(load_jsonl(Path(config.train_data_path))), tokenizer)
    eval_rows = None
    if config.eval_data_path is not None:
        eval_rows = _render_rows(build_sft_rows(load_jsonl(Path(config.eval_data_path))), tokenizer)

    train_dataset = Dataset.from_list(train_rows).map(
        lambda batch: _tokenize_batch(batch, tokenizer=tokenizer, max_length=config.max_length),
        batched=True,
        remove_columns=["text", "sample_id", "metadata"],
    )
    eval_dataset = None
    if eval_rows is not None:
        eval_dataset = Dataset.from_list(eval_rows).map(
            lambda batch: _tokenize_batch(batch, tokenizer=tokenizer, max_length=config.max_length),
            batched=True,
            remove_columns=["text", "sample_id", "metadata"],
        )

    model = load_causal_lm(
        model_name_or_path=config.model_name_or_path,
        adapter_config=adapter_config,
        precision_config=precision_config,
    )
    torch_dtype = resolve_torch_dtype(precision_config.torch_dtype)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.save_steps if eval_dataset is not None else None,
        bf16=torch_dtype is not None and str(torch_dtype).endswith("bfloat16"),
        fp16=torch_dtype is not None and str(torch_dtype).endswith("float16"),
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute an SFT training run from a generated config file."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the generated SFT config JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return execute_trl_sft_training(load_trl_sft_config(args.config_path))


def _render_rows(records: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "sample_id": str(record.get("sample_id", "")),
                "text": render_chat_messages(record["messages"], tokenizer),
                "metadata": dict(record.get("metadata", {}))
                if isinstance(record.get("metadata"), Mapping)
                else {},
            }
        )
    return rows


def _tokenize_batch(
    batch: Mapping[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    texts = [str(item) for item in batch.get("text", [])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = [list(input_ids) for input_ids in tokenized["input_ids"]]
    return tokenized


if __name__ == "__main__":
    raise SystemExit(main())
