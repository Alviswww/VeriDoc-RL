from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from veridoc_rl.predictions import parse_prediction_text
from veridoc_rl.rewards import score_verifier_results
from veridoc_rl.verifiers import run_verifier_suite


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Mapping[str, Any],
    extra_info: str | Mapping[str, Any] | None = None,
) -> float:
    del data_source
    reference = _coerce_mapping(ground_truth)
    extra_payload = _coerce_mapping(extra_info) if extra_info is not None else {}
    reward_profile = str(extra_payload.get("reward_profile", "rlvr"))
    context = _coerce_mapping(extra_payload.get("context")) if extra_payload.get("context") is not None else {}
    prediction = parse_prediction_text(
        solution_str,
        sample_id=str(reference.get("sample_id", "")),
    )
    results = run_verifier_suite(
        prediction=prediction,
        reference=reference,
        context=context,
    )
    reward = score_verifier_results(results, profile=reward_profile)
    return float(reward["total_reward"])


def _coerce_mapping(value: str | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}
