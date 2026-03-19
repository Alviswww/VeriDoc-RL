from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from veridoc_rl.evaluation import load_jsonl
from veridoc_rl.experiments import load_experiment_matrix
from veridoc_rl.predictions import parse_prediction_text
from veridoc_rl.training.prompting import DEFAULT_SYSTEM_PROMPT, build_chat_messages


@dataclass(slots=True, frozen=True)
class CandidateGenerationConfig:
    model: str
    backend: str = "vllm"
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    num_candidates: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024
    timeout_seconds: int = 120
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_candidates_for_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: CandidateGenerationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        input_payload, metadata = _extract_input_record(record)
        sample_id = str(input_payload.get("sample_id", ""))
        response_choices = request_chat_candidates(input_payload=input_payload, config=config)
        for index, content in enumerate(response_choices):
            rows.append(
                {
                    "candidate_id": f"{sample_id}::cand_{index}",
                    "sample_id": sample_id,
                    "backend": config.backend,
                    "model": config.model,
                    "generation_config": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "max_new_tokens": config.max_new_tokens,
                        "num_candidates": config.num_candidates,
                    },
                    "metadata": metadata,
                    "prediction": parse_prediction_text(content, sample_id=sample_id),
                    "raw_text": content,
                }
            )
    return rows


def export_candidate_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def request_chat_candidates(
    *,
    input_payload: Mapping[str, Any],
    config: CandidateGenerationConfig,
) -> list[str]:
    if config.backend != "vllm":
        raise ValueError(f"Unsupported inference backend: {config.backend}")
    messages = build_chat_messages(
        input_payload,
        system_prompt=config.system_prompt,
    )
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
        "n": config.num_candidates,
    }
    response_payload = _post_json(
        url=f"{config.api_base.rstrip('/')}/chat/completions",
        payload=payload,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )
    choices = response_payload.get("choices", [])
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise RuntimeError("Inference response is missing `choices`.")
    rendered: list[str] = []
    for item in choices:
        if not isinstance(item, Mapping):
            continue
        message = item.get("message")
        if not isinstance(message, Mapping):
            continue
        rendered.append(str(message.get("content", "")).strip())
    if not rendered:
        raise RuntimeError("Inference response did not contain any candidate completions.")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate candidate predictions through a vLLM OpenAI-compatible API."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Candidate JSONL output.")
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=Path("configs/experiment_matrix.yaml"),
        help="Optional experiment matrix used to populate defaults.",
    )
    parser.add_argument("--model", help="Override model name used for candidate generation.")
    parser.add_argument(
        "--backend",
        choices=("vllm",),
        help="Inference backend. Defaults to the matrix inference backend.",
    )
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the inference server.")
    parser.add_argument("--num-candidates", type=int, help="Number of candidates per sample.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, help="Top-p sampling cutoff.")
    parser.add_argument("--max-new-tokens", type=int, help="Maximum completion tokens.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override system prompt used for generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    defaults = _load_generation_defaults(args.matrix_path)
    config = CandidateGenerationConfig(
        model=args.model or defaults["model"],
        backend=args.backend or str(defaults["backend"]),
        api_base=args.api_base or str(defaults["api_base"]),
        api_key=args.api_key,
        num_candidates=int(
            args.num_candidates if args.num_candidates is not None else defaults["num_candidates"]
        ),
        temperature=float(
            args.temperature if args.temperature is not None else defaults["temperature"]
        ),
        top_p=float(args.top_p if args.top_p is not None else defaults["top_p"]),
        max_new_tokens=int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else defaults["max_new_tokens"]
        ),
        timeout_seconds=int(args.timeout_seconds),
        system_prompt=args.system_prompt,
    )
    export_candidate_jsonl(
        args.output_path,
        generate_candidates_for_records(load_jsonl(args.input_path), config=config),
    )
    return 0


def _extract_input_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = record.get("input")
    if isinstance(input_payload, Mapping):
        metadata = record.get("metadata")
        return dict(input_payload), dict(metadata) if isinstance(metadata, Mapping) else {}
    return dict(record), {}


def _load_generation_defaults(matrix_path: Path) -> dict[str, Any]:
    matrix = load_experiment_matrix(matrix_path)
    inference = dict(matrix.inference)
    return {
        "model": matrix.base_model.get("mvp", "Qwen/Qwen3.5-0.8B"),
        "backend": str(inference.get("backend", "vllm")),
        "api_base": str(inference.get("api_base", "http://127.0.0.1:8000/v1")),
        "num_candidates": int(inference.get("num_candidates", 4)),
        "temperature": float(inference.get("temperature", 0.8)),
        "top_p": float(inference.get("top_p", 0.95)),
        "max_new_tokens": int(inference.get("max_new_tokens", 1024)),
    }


def _post_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    api_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach inference server at {url}: {exc.reason}") from exc

    payload_obj = json.loads(raw)
    if not isinstance(payload_obj, Mapping):
        raise RuntimeError("Inference response must be a JSON object.")
    return dict(payload_obj)


if __name__ == "__main__":
    raise SystemExit(main())
