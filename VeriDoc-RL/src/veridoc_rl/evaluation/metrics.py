from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from veridoc_rl.normalizers import normalize_known_field
from veridoc_rl.schema import validate_prediction_payload


def canonicalize_field_value(field_name: str, value: Any) -> Any:
    """Normalize a field value for metric comparisons when possible."""

    normalized = normalize_known_field(field_name, value)
    if normalized is not None:
        return normalized
    if isinstance(value, Mapping):
        return {key: canonicalize_field_value(key, val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [canonicalize_field_value(field_name, item) for item in value]
    return value


def compute_field_level_metrics(
    prediction_fields: Mapping[str, Any],
    reference_fields: Mapping[str, Any],
) -> dict[str, float]:
    predicted_keys = set(prediction_fields)
    reference_keys = set(reference_fields)
    true_positive = 0
    for key in predicted_keys & reference_keys:
        if canonicalize_field_value(key, prediction_fields[key]) == canonicalize_field_value(
            key,
            reference_fields[key],
        ):
            true_positive += 1

    false_positive = len(predicted_keys - reference_keys) + sum(
        1
        for key in predicted_keys & reference_keys
        if canonicalize_field_value(key, prediction_fields[key])
        != canonicalize_field_value(key, reference_fields[key])
    )
    false_negative = len(reference_keys - predicted_keys) + sum(
        1
        for key in predicted_keys & reference_keys
        if canonicalize_field_value(key, prediction_fields[key])
        != canonicalize_field_value(key, reference_fields[key])
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_form_exact_match(
    prediction_fields: Mapping[str, Any],
    reference_fields: Mapping[str, Any],
) -> float:
    return float(
        all(
            canonicalize_field_value(key, prediction_fields.get(key))
            == canonicalize_field_value(key, reference_fields.get(key))
            for key in sorted(set(prediction_fields) | set(reference_fields))
        )
    )


def compute_rule_pass_rate(validations: list[Mapping[str, Any]]) -> float:
    applicable = [item for item in validations if item.get("status") != "not_applicable"]
    if not applicable:
        return 1.0
    passed = sum(1 for item in applicable if item.get("status") == "pass")
    return passed / len(applicable)


def compute_invalid_json_rate(items: Iterable[str | Mapping[str, Any]]) -> float:
    payloads = list(items)
    if not payloads:
        return 0.0

    invalid = 0
    for item in payloads:
        if isinstance(item, Mapping):
            if validate_prediction_payload(item):
                invalid += 1
            continue
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(payload, Mapping) or validate_prediction_payload(payload):
            invalid += 1
    return invalid / len(payloads)


def compute_validation_match_rate(
    prediction_validations: list[Mapping[str, Any]],
    reference_validations: list[Mapping[str, Any]],
) -> float:
    """Measure rule-status agreement between prediction and reference outputs."""

    prediction_statuses = {
        str(item["rule_id"]): str(item.get("status"))
        for item in prediction_validations
        if "rule_id" in item
    }
    reference_statuses = {
        str(item["rule_id"]): str(item.get("status"))
        for item in reference_validations
        if "rule_id" in item
    }
    all_rule_ids = set(prediction_statuses) | set(reference_statuses)
    if not all_rule_ids:
        return 1.0

    matched = sum(
        1
        for rule_id in all_rule_ids
        if prediction_statuses.get(rule_id) == reference_statuses.get(rule_id)
    )
    return matched / len(all_rule_ids)


def evaluate_prediction(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, float]:
    prediction_fields = prediction.get("fields", {})
    reference_fields = reference.get("fields", {})
    if not isinstance(prediction_fields, Mapping) or not isinstance(reference_fields, Mapping):
        raise ValueError("Both prediction and reference must expose a 'fields' mapping.")

    validations = prediction.get("validations", [])
    if not isinstance(validations, list):
        raise ValueError("Prediction 'validations' must be a list.")
    reference_validations = reference.get("validations", [])
    if not isinstance(reference_validations, list):
        raise ValueError("Reference 'validations' must be a list.")

    field_metrics = compute_field_level_metrics(prediction_fields, reference_fields)
    return {
        "field_precision": field_metrics["precision"],
        "field_recall": field_metrics["recall"],
        "field_f1": field_metrics["f1"],
        "form_exact_match": compute_form_exact_match(prediction_fields, reference_fields),
        "rule_pass_rate": compute_rule_pass_rate(validations),
        "validation_match_rate": compute_validation_match_rate(validations, reference_validations),
        "invalid_json_rate": compute_invalid_json_rate([prediction]),
    }
