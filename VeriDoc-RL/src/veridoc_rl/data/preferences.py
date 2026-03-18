from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation import evaluate_prediction, load_jsonl
from veridoc_rl.rewards import list_reward_profiles, score_verifier_results
from veridoc_rl.verifiers import BaseVerifier, VerificationResult, run_verifier_suite


@dataclass(slots=True)
class PreferenceExample:
    sample_id: str
    input_payload: dict[str, Any] | None
    reference: dict[str, Any]
    metadata: dict[str, Any]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    chosen_reward: dict[str, Any]
    rejected_reward: dict[str, Any]
    reward_margin: float
    reward_profile: str
    chosen_candidate_id: str
    rejected_candidate_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "task_type": "DPO_preference",
            "sample_id": self.sample_id,
            "reward_profile": self.reward_profile,
            "reward_margin": self.reward_margin,
            "metadata": dict(self.metadata),
            "input": self.input_payload,
            "reference": self.reference,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
            "chosen_candidate_id": self.chosen_candidate_id,
            "rejected_candidate_id": self.rejected_candidate_id,
        }


@dataclass(slots=True)
class ScoredCandidate:
    sample_id: str
    candidate_id: str
    prediction: dict[str, Any]
    reward: dict[str, Any]
    metrics: dict[str, float]
    verifier_results: list[VerificationResult]


def build_preference_pairs(
    candidate_records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    *,
    reward_profile: str = "default",
    min_margin: float = 0.0,
    include_all_pairs: bool = False,
    max_pairs_per_sample: int = 1,
    verifiers: Sequence[BaseVerifier] | None = None,
) -> list[PreferenceExample]:
    if max_pairs_per_sample <= 0:
        return []

    indexed_references = _index_reference_records(reference_records)
    grouped_candidates: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        prediction = _unwrap_payload(record, key="prediction")
        sample_id = _extract_sample_id(prediction, source="candidate")
        grouped_candidates[sample_id].append(record)

    missing_references = sorted(set(grouped_candidates) - set(indexed_references))
    if missing_references:
        raise ValueError(
            "Missing references for candidate sample_ids: " + ", ".join(missing_references)
        )

    examples: list[PreferenceExample] = []
    for sample_id in sorted(grouped_candidates):
        reference_bundle = indexed_references[sample_id]
        scored_candidates = _score_candidates(
            sample_id=sample_id,
            candidate_records=grouped_candidates[sample_id],
            reference_bundle=reference_bundle,
            reward_profile=reward_profile,
            verifiers=verifiers,
        )
        if len(scored_candidates) < 2:
            continue

        sample_examples = _build_sample_pairs(
            scored_candidates,
            reference_bundle=reference_bundle,
            reward_profile=reward_profile,
            min_margin=min_margin,
            include_all_pairs=include_all_pairs,
            max_pairs=max_pairs_per_sample,
        )
        examples.extend(sample_examples)
    return examples


def export_preference_jsonl(path: Path, examples: Sequence[PreferenceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs from candidate outputs.")
    parser.add_argument("--reference-path", type=Path, required=True, help="Reference JSONL path.")
    parser.add_argument("--candidate-path", type=Path, required=True, help="Candidate JSONL path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Preference JSONL output path.")
    parser.add_argument(
        "--reward-profile",
        choices=list_reward_profiles(),
        default="default",
        help="Reward profile used to rank candidates.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Minimum reward margin required to keep a chosen/rejected pair.",
    )
    parser.add_argument(
        "--max-pairs-per-sample",
        type=int,
        default=1,
        help="Maximum number of preference pairs exported per sample.",
    )
    parser.add_argument(
        "--include-all-pairs",
        action="store_true",
        help="Export multiple ranked pairs per sample instead of best-vs-worst only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    examples = build_preference_pairs(
        candidate_records=load_jsonl(args.candidate_path),
        reference_records=load_jsonl(args.reference_path),
        reward_profile=args.reward_profile,
        min_margin=args.min_margin,
        include_all_pairs=args.include_all_pairs,
        max_pairs_per_sample=args.max_pairs_per_sample,
    )
    export_preference_jsonl(args.output_path, examples)
    return 0


def _index_reference_records(reference_records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed_references: dict[str, dict[str, Any]] = {}
    for record in reference_records:
        reference = _unwrap_payload(record, key="reference")
        sample_id = _extract_sample_id(reference, source="reference")
        if sample_id in indexed_references:
            raise ValueError(f"Duplicate reference sample_id: {sample_id}")
        indexed_references[sample_id] = {
            "reference": dict(reference),
            "metadata": dict(_optional_mapping(record.get("metadata")) or {}),
            "input": dict(_optional_mapping(record.get("input")) or {}) or None,
            "context": dict(_optional_mapping(record.get("context")) or {}),
        }
    return indexed_references


def _score_candidates(
    *,
    sample_id: str,
    candidate_records: Sequence[Mapping[str, Any]],
    reference_bundle: Mapping[str, Any],
    reward_profile: str,
    verifiers: Sequence[BaseVerifier] | None,
) -> list[ScoredCandidate]:
    reference = _require_mapping(reference_bundle, "reference")
    base_context = dict(_optional_mapping(reference_bundle.get("context")) or {})
    scored: list[ScoredCandidate] = []
    for index, record in enumerate(candidate_records):
        prediction = dict(_unwrap_payload(record, key="prediction"))
        context = dict(base_context)
        context.update(dict(_optional_mapping(record.get("context")) or {}))
        verifier_results = run_verifier_suite(
            prediction=prediction,
            reference=reference,
            context=context,
            verifiers=verifiers,
        )
        reward = score_verifier_results(verifier_results, profile=reward_profile)
        metrics = evaluate_prediction(prediction=prediction, reference=reference)
        metrics["total_reward"] = float(reward["total_reward"])
        candidate_id = str(record.get("candidate_id") or f"{sample_id}::candidate_{index}")
        scored.append(
            ScoredCandidate(
                sample_id=sample_id,
                candidate_id=candidate_id,
                prediction=prediction,
                reward=reward,
                metrics=metrics,
                verifier_results=verifier_results,
            )
        )
    return sorted(scored, key=_candidate_sort_key, reverse=True)


def _build_sample_pairs(
    scored_candidates: Sequence[ScoredCandidate],
    *,
    reference_bundle: Mapping[str, Any],
    reward_profile: str,
    min_margin: float,
    include_all_pairs: bool,
    max_pairs: int,
) -> list[PreferenceExample]:
    pairs: list[PreferenceExample] = []
    if include_all_pairs:
        for higher_index, chosen in enumerate(scored_candidates):
            for rejected in scored_candidates[higher_index + 1 :]:
                pair = _build_pair(
                    chosen=chosen,
                    rejected=rejected,
                    reference_bundle=reference_bundle,
                    reward_profile=reward_profile,
                    min_margin=min_margin,
                )
                if pair is None:
                    continue
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    pair = _build_pair(
        chosen=scored_candidates[0],
        rejected=scored_candidates[-1],
        reference_bundle=reference_bundle,
        reward_profile=reward_profile,
        min_margin=min_margin,
    )
    if pair is not None:
        pairs.append(pair)
    return pairs


def _build_pair(
    *,
    chosen: ScoredCandidate,
    rejected: ScoredCandidate,
    reference_bundle: Mapping[str, Any],
    reward_profile: str,
    min_margin: float,
) -> PreferenceExample | None:
    reward_margin = float(chosen.reward["total_reward"]) - float(rejected.reward["total_reward"])
    if reward_margin < min_margin:
        return None
    return PreferenceExample(
        sample_id=chosen.sample_id,
        input_payload=dict(_optional_mapping(reference_bundle.get("input")) or {}) or None,
        reference=dict(_require_mapping(reference_bundle, "reference")),
        metadata=dict(_optional_mapping(reference_bundle.get("metadata")) or {}),
        chosen={
            "prediction": chosen.prediction,
            "metrics": chosen.metrics,
            "verifier_results": [_serialize_verifier_result(item) for item in chosen.verifier_results],
        },
        rejected={
            "prediction": rejected.prediction,
            "metrics": rejected.metrics,
            "verifier_results": [_serialize_verifier_result(item) for item in rejected.verifier_results],
        },
        chosen_reward=chosen.reward,
        rejected_reward=rejected.reward,
        reward_margin=reward_margin,
        reward_profile=reward_profile,
        chosen_candidate_id=chosen.candidate_id,
        rejected_candidate_id=rejected.candidate_id,
    )


def _candidate_sort_key(candidate: ScoredCandidate) -> tuple[float, float, float, float]:
    return (
        float(candidate.reward["total_reward"]),
        candidate.metrics.get("field_f1", 0.0),
        candidate.metrics.get("validation_match_rate", 0.0),
        candidate.metrics.get("form_exact_match", 0.0),
    )


def _serialize_verifier_result(result: VerificationResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "passed": result.passed,
        "score": result.score,
        "details": result.details,
    }


def _unwrap_payload(record: Mapping[str, Any], *, key: str) -> Mapping[str, Any]:
    payload = record.get(key, record)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{key} payload must be a mapping.")
    return payload


def _extract_sample_id(payload: Mapping[str, Any], *, source: str) -> str:
    sample_id = payload.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{source} payload must contain a non-empty string sample_id.")
    return sample_id


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _require_mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    payload = record.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Record is missing mapping field: {key}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
