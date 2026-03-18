from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation.metrics import canonicalize_field_value, evaluate_prediction
from veridoc_rl.rewards import list_reward_profiles, score_verifier_results
from veridoc_rl.verifiers import BaseVerifier, VerificationResult, run_verifier_suite


BUCKET_DIMENSIONS: tuple[str, ...] = (
    "template_family",
    "ocr_noise_level",
    "hard_case_type",
    "rule_complexity",
)
ERROR_TAXONOMY: dict[str, str] = {
    "missing_field": "漏抽字段",
    "incorrect_value": "值错误",
    "normalization_error": "标准化错误",
    "relation_error": "关系字段错误",
    "checkbox_logic_error": "勾选逻辑错误",
    "rule_misjudgment": "规则误判",
    "ocr_noise_vulnerability": "OCR 噪声脆弱",
    "json_schema_error": "JSON/schema 非法",
}


@dataclass(slots=True)
class CaseAnalysis:
    """Single-case evaluation output used for summaries and case export."""

    sample_id: str
    metrics: dict[str, float]
    bucket: dict[str, str]
    taxonomy: list[str]
    verifier_results: list[VerificationResult]
    prediction: dict[str, Any]
    reference: dict[str, Any]
    reward: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_payload: dict[str, Any] | None = None
    missing_fields: list[str] = field(default_factory=list)
    mismatched_fields: list[str] = field(default_factory=list)
    validation_mismatches: list[str] = field(default_factory=list)

    @property
    def failed_verifiers(self) -> list[VerificationResult]:
        return [result for result in self.verifier_results if not result.passed]

    @property
    def is_failure(self) -> bool:
        return bool(
            self.taxonomy
            or self.failed_verifiers
            or self.metrics.get("form_exact_match", 1.0) < 1.0
            or self.metrics.get("validation_match_rate", 1.0) < 1.0
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "bucket": dict(self.bucket),
            "metrics": dict(self.metrics),
            "taxonomy": list(self.taxonomy),
            "reward": dict(self.reward),
            "missing_fields": list(self.missing_fields),
            "mismatched_fields": list(self.mismatched_fields),
            "validation_mismatches": list(self.validation_mismatches),
            "verifier_results": [_serialize_verification_result(item) for item in self.verifier_results],
            "failed_verifiers": [item.name for item in self.failed_verifiers],
            "prediction": self.prediction,
            "reference": self.reference,
            "metadata": dict(self.metadata),
            "input": self.input_payload,
        }


@dataclass(slots=True)
class DatasetEvaluationReport:
    """Aggregated phase-a report with overall, bucket and failure-case views."""

    overall: dict[str, Any]
    bucket_metrics: dict[str, dict[str, dict[str, Any]]]
    error_taxonomy: dict[str, dict[str, Any]]
    failure_cases: list[CaseAnalysis]
    cases: list[CaseAnalysis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": dict(self.overall),
            "bucket_metrics": self.bucket_metrics,
            "error_taxonomy": self.error_taxonomy,
            "failure_cases": [case.to_record() for case in self.failure_cases],
        }


def evaluate_case(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    input_payload: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    verifiers: Sequence[BaseVerifier] | None = None,
    reward_profile: str = "default",
) -> CaseAnalysis:
    """Evaluate one prediction/reference pair and attach verifier diagnostics."""

    prediction_payload = dict(prediction)
    reference_payload = dict(reference)
    metrics = evaluate_prediction(prediction_payload, reference_payload)
    verifier_results = run_verifier_suite(
        prediction=prediction_payload,
        reference=reference_payload,
        context=context,
        verifiers=verifiers,
    )
    reward = score_verifier_results(verifier_results, profile=reward_profile)
    metrics["verifier_pass_rate"] = _compute_verifier_pass_rate(verifier_results)
    metrics["total_reward"] = float(reward["total_reward"])

    prediction_fields = prediction_payload.get("fields")
    reference_fields = reference_payload.get("fields")
    missing_fields, mismatched_fields = _diff_fields(prediction_fields, reference_fields)
    validation_mismatches = _diff_validations(
        prediction_payload.get("validations"),
        reference_payload.get("validations"),
    )
    taxonomy = _classify_error_taxonomy(
        verifier_results=verifier_results,
        missing_fields=missing_fields,
        mismatched_fields=mismatched_fields,
        validation_mismatches=validation_mismatches,
    )
    sample_id = str(
        prediction_payload.get("sample_id")
        or reference_payload.get("sample_id")
        or (metadata or {}).get("sample_id")
        or ""
    )

    return CaseAnalysis(
        sample_id=sample_id,
        metrics=metrics,
        bucket=_extract_bucket(metadata),
        taxonomy=taxonomy,
        verifier_results=verifier_results,
        reward=reward,
        prediction=prediction_payload,
        reference=reference_payload,
        metadata=dict(metadata or {}),
        input_payload=dict(input_payload) if isinstance(input_payload, Mapping) else None,
        missing_fields=missing_fields,
        mismatched_fields=mismatched_fields,
        validation_mismatches=validation_mismatches,
    )


def evaluate_dataset(
    entries: Sequence[Mapping[str, Any]],
    *,
    verifiers: Sequence[BaseVerifier] | None = None,
    failure_case_limit: int = 3,
    reward_profile: str = "default",
) -> DatasetEvaluationReport:
    """Run the full phase-a reporting pipeline over aligned evaluation entries."""

    cases = [
        evaluate_case(
            prediction=_require_mapping(entry, "prediction"),
            reference=_require_mapping(entry, "reference"),
            metadata=_optional_mapping(entry.get("metadata")),
            input_payload=_optional_mapping(entry.get("input")),
            context=_optional_mapping(entry.get("context")),
            verifiers=verifiers,
            reward_profile=reward_profile,
        )
        for entry in entries
    ]
    failure_cases = _select_failure_cases(cases, limit=failure_case_limit)
    return DatasetEvaluationReport(
        overall=_summarize_cases(cases),
        bucket_metrics=_summarize_bucket_metrics(cases),
        error_taxonomy=_summarize_error_taxonomy(cases),
        failure_cases=failure_cases,
        cases=cases,
    )

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of mapping records."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object per line.")
            rows.append(dict(payload))
    return rows


def build_evaluation_entries(
    prediction_records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align prediction JSONL rows with reference dataset rows by sample_id."""

    indexed_references: dict[str, dict[str, Any]] = {}
    for record in reference_records:
        reference_payload = _unwrap_payload(record, key="reference")
        sample_id = _extract_sample_id(reference_payload, source="reference")
        if sample_id in indexed_references:
            raise ValueError(f"Duplicate reference sample_id: {sample_id}")
        indexed_references[sample_id] = {
            "reference": dict(reference_payload),
            "metadata": dict(_optional_mapping(record.get("metadata")) or {}),
            "input": dict(_optional_mapping(record.get("input")) or {}) or None,
            "context": dict(_optional_mapping(record.get("context")) or {}),
        }

    entries: list[dict[str, Any]] = []
    seen_prediction_ids: set[str] = set()
    for record in prediction_records:
        prediction_payload = _unwrap_payload(record, key="prediction")
        sample_id = _extract_sample_id(prediction_payload, source="prediction")
        if sample_id in seen_prediction_ids:
            raise ValueError(f"Duplicate prediction sample_id: {sample_id}")
        seen_prediction_ids.add(sample_id)
        if sample_id not in indexed_references:
            raise ValueError(f"Missing reference for prediction sample_id: {sample_id}")

        reference_bundle = indexed_references.pop(sample_id)
        metadata = dict(reference_bundle["metadata"])
        metadata.update(dict(_optional_mapping(record.get("metadata")) or {}))
        context = dict(reference_bundle["context"])
        context.update(dict(_optional_mapping(record.get("context")) or {}))
        input_payload = reference_bundle["input"]
        if input_payload is None:
            input_payload = dict(_optional_mapping(record.get("input")) or {}) or None

        entries.append(
            {
                "prediction": dict(prediction_payload),
                "reference": dict(reference_bundle["reference"]),
                "metadata": metadata,
                "input": input_payload,
                "context": context,
            }
        )

    if indexed_references:
        missing_ids = sorted(indexed_references)
        raise ValueError(
            "Missing predictions for reference sample_ids: " + ", ".join(missing_ids)
        )
    return entries


def evaluate_dataset_files(
    reference_path: Path,
    prediction_path: Path,
    *,
    failure_case_limit: int = 3,
    verifiers: Sequence[BaseVerifier] | None = None,
    reward_profile: str = "default",
) -> DatasetEvaluationReport:
    """Load JSONL inputs from disk and produce a dataset evaluation report."""

    reference_records = load_jsonl(reference_path)
    prediction_records = load_jsonl(prediction_path)
    entries = build_evaluation_entries(prediction_records, reference_records)
    return evaluate_dataset(
        entries,
        verifiers=verifiers,
        failure_case_limit=failure_case_limit,
        reward_profile=reward_profile,
    )


def export_case_records(
    path: Path,
    cases: Iterable[CaseAnalysis],
    *,
    failures_only: bool = False,
) -> None:
    """Export case records as JSONL for downstream case-study analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            if failures_only and not case.is_failure:
                continue
            handle.write(json.dumps(case.to_record(), ensure_ascii=False) + "\n")


def write_report(path: Path, report: DatasetEvaluationReport) -> None:
    """Persist the aggregate report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase-a evaluation and reporting.")
    parser.add_argument("--reference-path", type=Path, required=True, help="Reference JSONL path.")
    parser.add_argument("--prediction-path", type=Path, required=True, help="Prediction JSONL path.")
    parser.add_argument("--report-path", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument(
        "--case-export-path",
        type=Path,
        help="Optional JSONL path for exporting per-case diagnostics.",
    )
    parser.add_argument(
        "--failure-only",
        action="store_true",
        help="Export only failed cases when --case-export-path is provided.",
    )
    parser.add_argument(
        "--failure-case-limit",
        type=int,
        default=3,
        help="Number of failure cases to keep in the summary report.",
    )
    parser.add_argument(
        "--reward-profile",
        choices=list_reward_profiles(),
        default="default",
        help="Reward profile used for composite reward scoring and ablations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate_dataset_files(
        reference_path=args.reference_path,
        prediction_path=args.prediction_path,
        failure_case_limit=args.failure_case_limit,
        reward_profile=args.reward_profile,
    )
    write_report(args.report_path, report)
    if args.case_export_path is not None:
        export_case_records(args.case_export_path, report.cases, failures_only=args.failure_only)
    return 0


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
        raise ValueError(f"Evaluation entry is missing mapping field: {key}")
    return payload


def _extract_bucket(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    bucket_source = metadata.get("bucket") if isinstance(metadata, Mapping) else None
    bucket = bucket_source if isinstance(bucket_source, Mapping) else {}
    return {
        dimension: str(bucket.get(dimension, "unknown"))
        for dimension in BUCKET_DIMENSIONS
    }


def _diff_fields(
    prediction_fields: Any,
    reference_fields: Any,
) -> tuple[list[str], list[str]]:
    prediction_mapping = prediction_fields if isinstance(prediction_fields, Mapping) else {}
    reference_mapping = reference_fields if isinstance(reference_fields, Mapping) else {}

    missing_fields = [
        key
        for key in sorted(reference_mapping)
        if key not in prediction_mapping or _is_missing_value(prediction_mapping.get(key))
    ]
    mismatched_fields = [
        key
        for key in sorted(set(prediction_mapping) & set(reference_mapping))
        if key not in missing_fields
        and canonicalize_field_value(key, prediction_mapping.get(key))
        != canonicalize_field_value(key, reference_mapping.get(key))
    ]
    return missing_fields, mismatched_fields


def _diff_validations(
    prediction_validations: Any,
    reference_validations: Any,
) -> list[str]:
    prediction_statuses = _collect_validation_statuses(prediction_validations)
    reference_statuses = _collect_validation_statuses(reference_validations)
    return [
        rule_id
        for rule_id in sorted(set(prediction_statuses) | set(reference_statuses))
        if prediction_statuses.get(rule_id) != reference_statuses.get(rule_id)
    ]


def _collect_validation_statuses(validations: Any) -> dict[str, str]:
    if not isinstance(validations, list):
        return {}
    return {
        str(item["rule_id"]): str(item.get("status"))
        for item in validations
        if isinstance(item, Mapping) and "rule_id" in item
    }


def _classify_error_taxonomy(
    *,
    verifier_results: Sequence[VerificationResult],
    missing_fields: Sequence[str],
    mismatched_fields: Sequence[str],
    validation_mismatches: Sequence[str],
) -> list[str]:
    verifier_map = {result.name: result for result in verifier_results}
    categories: list[str] = []

    schema_result = verifier_map.get("schema_reward")
    if schema_result is not None and not schema_result.passed:
        categories.append("json_schema_error")
    if missing_fields:
        categories.append("missing_field")
    if mismatched_fields:
        categories.append("incorrect_value")

    normalization_result = verifier_map.get("normalization_reward")
    if normalization_result is not None and normalization_result.details.get("invalid_fields"):
        categories.append("normalization_error")

    cross_field_result = verifier_map.get("cross_field_consistency_reward")
    failed_cross_checks = set(
        cross_field_result.details.get("failed_checks", []) if cross_field_result else []
    )
    if "policyholder_insured_relation" in failed_cross_checks:
        categories.append("relation_error")

    checkbox_result = verifier_map.get("checkbox_logic_reward")
    if checkbox_result is not None and checkbox_result.details.get("failed_checks"):
        categories.append("checkbox_logic_error")
    if validation_mismatches:
        categories.append("rule_misjudgment")

    ocr_result = verifier_map.get("ocr_robustness_reward")
    if ocr_result is not None and not ocr_result.passed:
        categories.append("ocr_noise_vulnerability")
    return categories


def _summarize_cases(cases: Sequence[CaseAnalysis]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(cases),
        "failure_count": sum(1 for case in cases if case.is_failure),
        "verifier_scores": {},
    }
    if not cases:
        return summary

    metric_keys = sorted({metric for case in cases for metric in case.metrics})
    for metric_key in metric_keys:
        summary[metric_key] = sum(case.metrics.get(metric_key, 0.0) for case in cases) / len(cases)
    summary["verifier_scores"] = _average_verifier_scores(cases)
    return summary


def _average_verifier_scores(cases: Sequence[CaseAnalysis]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for result in case.verifier_results:
            scores[result.name].append(result.score)
    return {
        name: sum(values) / len(values)
        for name, values in sorted(scores.items())
        if values
    }


def _summarize_bucket_metrics(
    cases: Sequence[CaseAnalysis],
) -> dict[str, dict[str, dict[str, Any]]]:
    bucket_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in BUCKET_DIMENSIONS:
        grouped: dict[str, list[CaseAnalysis]] = defaultdict(list)
        for case in cases:
            grouped[case.bucket.get(dimension, "unknown")].append(case)
        bucket_metrics[dimension] = {
            bucket_name: _summarize_cases(bucket_cases)
            for bucket_name, bucket_cases in sorted(grouped.items())
        }
    return bucket_metrics


def _summarize_error_taxonomy(cases: Sequence[CaseAnalysis]) -> dict[str, dict[str, Any]]:
    total_cases = len(cases)
    counts = {category: 0 for category in ERROR_TAXONOMY}
    for case in cases:
        for category in case.taxonomy:
            counts[category] += 1

    return {
        category: {
            "label": label,
            "count": counts[category],
            "share": 0.0 if total_cases == 0 else counts[category] / total_cases,
        }
        for category, label in ERROR_TAXONOMY.items()
    }


def _select_failure_cases(cases: Sequence[CaseAnalysis], *, limit: int) -> list[CaseAnalysis]:
    failures = [case for case in cases if case.is_failure]
    return sorted(failures, key=_failure_sort_key)[: max(limit, 0)]


def _failure_sort_key(case: CaseAnalysis) -> tuple[float, float, float, float, int, str]:
    return (
        case.metrics.get("form_exact_match", 1.0),
        case.metrics.get("field_f1", 1.0),
        case.metrics.get("validation_match_rate", 1.0),
        case.metrics.get("verifier_pass_rate", 1.0),
        -len(case.taxonomy),
        case.sample_id,
    )


def _compute_verifier_pass_rate(results: Sequence[VerificationResult]) -> float:
    if not results:
        return 1.0
    return sum(1 for result in results if result.passed) / len(results)


def _serialize_verification_result(result: VerificationResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "passed": result.passed,
        "score": result.score,
        "details": result.details,
    }


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


if __name__ == "__main__":
    raise SystemExit(main())
