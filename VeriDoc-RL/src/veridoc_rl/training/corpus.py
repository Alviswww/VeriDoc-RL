from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import load_jsonl
from veridoc_rl.training.prompting import (
    DEFAULT_SYSTEM_PROMPT,
    build_assistant_response,
    build_chat_messages,
    build_user_prompt,
)


SUPPORTED_TRAINING_STAGES: tuple[str, ...] = ("phase_a_sft", "phase_b_dpo", "phase_c_rlvr")


def prepare_sft_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        input_payload = _require_mapping(record, "input")
        reference_payload = _require_mapping(record, "reference")
        prepared.append(
            {
                "task_type": str(record.get("task_type", "SFT_gold")),
                "stage": "phase_a_sft",
                "sample_id": str(reference_payload.get("sample_id", input_payload.get("sample_id", ""))),
                "messages": build_chat_messages(
                    input_payload,
                    reference_payload=reference_payload,
                    system_prompt=system_prompt,
                ),
                "metadata": dict(_optional_mapping(record.get("metadata")) or {}),
            }
        )
    return prepared


def prepare_dpo_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        input_payload = _require_mapping(record, "input")
        chosen_bundle = _require_mapping(record, "chosen")
        rejected_bundle = _require_mapping(record, "rejected")
        chosen_prediction = _require_mapping(chosen_bundle, "prediction")
        rejected_prediction = _require_mapping(rejected_bundle, "prediction")
        prompt = build_user_prompt(input_payload)
        prepared.append(
            {
                "task_type": "DPO_preference",
                "stage": "phase_b_dpo",
                "sample_id": str(record.get("sample_id", input_payload.get("sample_id", ""))),
                "system_prompt": system_prompt,
                "prompt": prompt,
                "chosen": build_assistant_response(chosen_prediction),
                "rejected": build_assistant_response(rejected_prediction),
                "chosen_candidate_id": str(record.get("chosen_candidate_id", "")),
                "rejected_candidate_id": str(record.get("rejected_candidate_id", "")),
                "reward_margin": float(record.get("reward_margin", 0.0)),
                "reward_profile": str(record.get("reward_profile", "default")),
                "metadata": dict(_optional_mapping(record.get("metadata")) or {}),
            }
        )
    return prepared


def prepare_rl_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reward_profile: str = "rlvr",
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        input_payload = _require_mapping(record, "input")
        payload: dict[str, Any] = {
            "task_type": str(record.get("task_type", "RL_prompt_only")),
            "stage": "phase_c_rlvr",
            "sample_id": str(input_payload.get("sample_id", "")),
            "system_prompt": system_prompt,
            "prompt": build_user_prompt(input_payload),
            "reward_profile": reward_profile,
            "metadata": dict(_optional_mapping(record.get("metadata")) or {}),
        }
        reference_payload = _optional_mapping(record.get("reference"))
        if reference_payload is not None:
            payload["reference"] = {
                "sample_id": reference_payload.get("sample_id"),
                "fields": reference_payload.get("fields", {}),
                "validations": reference_payload.get("validations", []),
            }
        prepared.append(payload)
    return prepared


def export_training_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare training corpora for SFT, DPO or RLVR stages.")
    parser.add_argument("--input-path", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--stage", choices=SUPPORTED_TRAINING_STAGES, required=True, help="Training stage to prepare.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Override system prompt.")
    parser.add_argument("--reward-profile", default="rlvr", help="Reward profile tag for RL corpora.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    records = load_jsonl(args.input_path)
    if args.stage == "phase_a_sft":
        prepared = prepare_sft_corpus(records, system_prompt=args.system_prompt)
    elif args.stage == "phase_b_dpo":
        prepared = prepare_dpo_corpus(records, system_prompt=args.system_prompt)
    else:
        prepared = prepare_rl_corpus(
            records,
            system_prompt=args.system_prompt,
            reward_profile=args.reward_profile,
        )
    export_training_jsonl(args.output_path, prepared)
    return 0


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _require_mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    payload = record.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Record is missing mapping field: {key}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
