from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import load_jsonl
from veridoc_rl.predictions import parse_prediction_text
from veridoc_rl.training.finetune import (
    AdapterConfig,
    PrecisionConfig,
    adapter_config_from_mapping,
    load_generation_model,
    load_tokenizer,
    precision_config_from_mapping,
    render_chat_messages,
)
from veridoc_rl.training.prompting import DEFAULT_SYSTEM_PROMPT, build_chat_messages


@dataclass(slots=True, frozen=True)
class InferenceConfig:
    model_name_or_path: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    adapter_config: dict[str, Any] | None = None
    precision_config: dict[str, Any] | None = None
    disable_thinking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_inference_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: InferenceConfig,
) -> list[dict[str, Any]]:
    adapter_config = adapter_config_from_mapping(config.adapter_config)
    precision_config = precision_config_from_mapping(config.precision_config)
    tokenizer = load_tokenizer(config.model_name_or_path)
    model = load_generation_model(
        model_name_or_path=config.model_name_or_path,
        adapter_config=adapter_config,
        precision_config=precision_config,
    )

    outputs: list[dict[str, Any]] = []
    for record in records:
        input_payload, metadata = _extract_input_record(record)
        generated_text = _generate_prediction_text(
            input_payload=input_payload,
            tokenizer=tokenizer,
            model=model,
            config=config,
        )
        sample_id = str(input_payload.get("sample_id", ""))
        outputs.append(
            {
                "sample_id": sample_id,
                "input": input_payload,
                "metadata": metadata,
                "prediction": parse_prediction_text(generated_text, sample_id=sample_id),
                "raw_text": generated_text,
                "model": config.model_name_or_path,
            }
        )
    return outputs


def export_prediction_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def select_first_candidate_predictions(
    candidate_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        sample_id = str(record.get("sample_id") or _extract_prediction_sample_id(record))
        grouped[sample_id].append(record)

    outputs: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        first = grouped[sample_id][0]
        prediction = first.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError(f"Candidate record for {sample_id} is missing `prediction`.")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "prediction": dict(prediction),
        }
        metadata = first.get("metadata")
        if isinstance(metadata, Mapping):
            row["metadata"] = dict(metadata)
        input_payload = first.get("input")
        if isinstance(input_payload, Mapping):
            row["input"] = dict(input_payload)
        outputs.append(row)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local checkpoint inference and export predictions as JSONL."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Prediction JSONL path.")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Baseline HF model name or local checkpoint path.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override system prompt used for generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--adapter-config",
        help="Optional JSON string overriding adapter config.",
    )
    parser.add_argument(
        "--precision-config",
        help="Optional JSON string overriding precision config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = run_inference_records(
        load_jsonl(args.input_path),
        config=InferenceConfig(
            model_name_or_path=args.model_name_or_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            adapter_config=_parse_optional_json(args.adapter_config),
            precision_config=_parse_optional_json(args.precision_config),
        ),
    )
    export_prediction_jsonl(args.output_path, rows)
    return 0


def _generate_prediction_text(
    *,
    input_payload: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    config: InferenceConfig,
) -> str:
    import torch

    messages = build_chat_messages(
        input_payload,
        system_prompt=config.system_prompt,
    )
    prompt_text = _render_generation_prompt(messages, tokenizer, disable_thinking=config.disable_thinking)
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    tokenized = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in tokenized.items()
    }
    prompt_length = int(tokenized["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p
    with torch.no_grad():
        output_ids = model.generate(**tokenized, **generation_kwargs)
    completion_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _render_generation_prompt(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    disable_thinking: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {}
            if disable_thinking:
                kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            return str(
                tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            )
        except Exception:
            pass
    return render_chat_messages(messages, tokenizer) + "\n\nASSISTANT:"


def _extract_input_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = record.get("input")
    if isinstance(input_payload, Mapping):
        metadata = record.get("metadata")
        return dict(input_payload), dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata = record.get("metadata")
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"metadata", "reference", "prediction", "task_type"}
    }
    return payload, dict(metadata) if isinstance(metadata, Mapping) else {}


def _extract_prediction_sample_id(record: Mapping[str, Any]) -> str:
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        return ""
    return str(prediction.get("sample_id", ""))


def _parse_optional_json(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a JSON object.")
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
