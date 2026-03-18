from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from veridoc_rl.normalizers import (
    extract_birth_date_from_id_number,
    normalize_checkbox_value,
    normalize_id_number,
    normalize_known_field,
)
from veridoc_rl.schema import validate_prediction_payload
from veridoc_rl.verifiers.base import BaseVerifier, VerificationResult


class SchemaVerifier(BaseVerifier):
    name = "schema_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        issues = validate_prediction_payload(prediction)
        passed = not issues
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            name=self.name,
            details={"issues": issues},
        )


class FieldMatchVerifier(BaseVerifier):
    name = "field_match_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        if reference is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "missing_reference"},
            )

        prediction_fields = prediction.get("fields")
        reference_fields = reference.get("fields")
        if not isinstance(prediction_fields, Mapping) or not isinstance(reference_fields, Mapping):
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        matched = 0
        total = 0
        mismatched: list[str] = []
        for field_name, expected_value in reference_fields.items():
            total += 1
            actual_value = prediction_fields.get(field_name)
            if _canonicalize_field_value(field_name, actual_value) == _canonicalize_field_value(
                field_name,
                expected_value,
            ):
                matched += 1
            else:
                mismatched.append(field_name)

        score = matched / total if total else 1.0
        return VerificationResult(
            passed=matched == total,
            score=score,
            name=self.name,
            details={"matched_fields": matched, "total_fields": total, "mismatched": mismatched},
        )


class NormalizationVerifier(BaseVerifier):
    name = "normalization_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        prediction_fields = prediction.get("fields")
        if not isinstance(prediction_fields, Mapping):
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        checked = 0
        normalized_ok = 0
        invalid_fields: list[str] = []
        for field_name, value in prediction_fields.items():
            normalized = normalize_known_field(field_name, value)
            if normalized is None:
                continue
            checked += 1
            if normalized == value:
                normalized_ok += 1
            else:
                invalid_fields.append(field_name)

        score = normalized_ok / checked if checked else 1.0
        return VerificationResult(
            passed=not invalid_fields,
            score=score,
            name=self.name,
            details={
                "checked_fields": checked,
                "normalized_fields": normalized_ok,
                "invalid_fields": invalid_fields,
            },
        )


class CrossFieldConsistencyVerifier(BaseVerifier):
    name = "cross_field_consistency_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        prediction_fields = prediction.get("fields")
        if not isinstance(prediction_fields, Mapping):
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        checks: dict[str, bool | None] = {
            "birth_date_vs_id_number": _check_birth_date_vs_id(prediction_fields),
            "beneficiary_ratio_sum": _check_beneficiary_ratio_sum(prediction_fields),
            "policyholder_insured_relation": _check_relation(prediction_fields, context),
            "product_payment_combo": _check_product_payment_combo(prediction_fields, context),
        }
        applicable = {name: passed for name, passed in checks.items() if passed is not None}
        passed_count = sum(1 for passed in applicable.values() if passed)
        score = passed_count / len(applicable) if applicable else 1.0
        failed = [name for name, passed in applicable.items() if not passed]
        return VerificationResult(
            passed=not failed,
            score=score,
            name=self.name,
            details={"checks": checks, "failed_checks": failed},
        )


class CheckboxLogicVerifier(BaseVerifier):
    name = "checkbox_logic_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        prediction_fields = prediction.get("fields")
        if not isinstance(prediction_fields, Mapping):
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        checks: dict[str, bool | None] = {
            "payment_mode_exclusive": _check_payment_mode_exclusive(prediction_fields),
            "auto_debit_requires_account": _check_auto_debit_account(prediction_fields),
        }
        applicable = {name: passed for name, passed in checks.items() if passed is not None}
        passed_count = sum(1 for passed in applicable.values() if passed)
        score = passed_count / len(applicable) if applicable else 1.0
        failed = [name for name, passed in applicable.items() if not passed]
        return VerificationResult(
            passed=not failed,
            score=score,
            name=self.name,
            details={"checks": checks, "failed_checks": failed},
        )


class OCRRobustnessVerifier(BaseVerifier):
    name = "ocr_robustness_reward"

    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        prediction_fields = prediction.get("fields")
        if not isinstance(prediction_fields, Mapping):
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        perturbed_predictions = list((context or {}).get("perturbed_predictions", []))
        if not perturbed_predictions:
            return VerificationResult(
                passed=True,
                score=1.0,
                name=self.name,
                details={"reason": "no_perturbed_predictions"},
            )

        stable = 0
        for item in perturbed_predictions:
            candidate_fields = item.get("fields", {}) if isinstance(item, Mapping) else {}
            if not isinstance(candidate_fields, Mapping):
                continue
            if _canonicalize_nested(prediction_fields) == _canonicalize_nested(candidate_fields):
                stable += 1
        score = stable / len(perturbed_predictions)
        return VerificationResult(
            passed=stable == len(perturbed_predictions),
            score=score,
            name=self.name,
            details={"stable_predictions": stable, "total_predictions": len(perturbed_predictions)},
        )


def _canonicalize_field_value(field_name: str, value: Any) -> Any:
    normalized = normalize_known_field(field_name, value)
    if normalized is not None:
        return normalized
    return _canonicalize_nested(value)


def _canonicalize_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_nested(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize_nested(item) for item in value]
    return value


def _check_birth_date_vs_id(fields: Mapping[str, Any]) -> bool | None:
    birth_date = fields.get("insured_birth_date") or fields.get("policyholder_birth_date")
    id_number = fields.get("insured_id_number") or fields.get("policyholder_id_number")
    normalized_id = normalize_id_number(id_number)
    normalized_birth = normalize_known_field("insured_birth_date", birth_date)
    if normalized_id is None or normalized_birth is None:
        return None
    return extract_birth_date_from_id_number(normalized_id) == normalized_birth


def _parse_ratio(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().rstrip("%")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _extract_beneficiary_ratios(fields: Mapping[str, Any]) -> list[float]:
    ratios: list[float] = []
    beneficiaries = fields.get("beneficiaries")
    if isinstance(beneficiaries, list):
        for item in beneficiaries:
            if isinstance(item, Mapping):
                ratio = _parse_ratio(item.get("ratio"))
                if ratio is not None:
                    ratios.append(ratio)
    direct_ratio = _parse_ratio(fields.get("beneficiary_ratio"))
    if direct_ratio is not None and not ratios:
        ratios.append(direct_ratio)
    return ratios


def _check_beneficiary_ratio_sum(fields: Mapping[str, Any]) -> bool | None:
    ratios = _extract_beneficiary_ratios(fields)
    if not ratios:
        return None
    return abs(sum(ratios) - 100.0) < 1e-6


def _check_relation(fields: Mapping[str, Any], context: Mapping[str, Any] | None) -> bool | None:
    relation = fields.get("relation_policyholder_to_insured")
    if relation is None:
        return None
    allowed_relations = {
        item.lower()
        for item in (context or {}).get(
            "allowed_relations",
            ["self", "spouse", "child", "parent", "employee", "other"],
        )
    }
    return str(relation).strip().lower() in allowed_relations


def _check_product_payment_combo(
    fields: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> bool | None:
    product_name = fields.get("product_name")
    payment_mode = normalize_known_field("payment_mode", fields.get("payment_mode"))
    payment_period = fields.get("payment_period_years")
    if product_name is None and payment_mode is None and payment_period is None:
        return None
    allowed_modes = {
        item.lower()
        for item in (context or {}).get(
            "allowed_payment_modes",
            ["annual", "semi_annual", "quarterly", "monthly", "single_premium"],
        )
    }
    if payment_mode is not None and payment_mode not in allowed_modes:
        return False
    if payment_mode == "single_premium":
        return payment_period in (None, 1, "1")
    return payment_period is None or str(payment_period).isdigit()


def _check_payment_mode_exclusive(fields: Mapping[str, Any]) -> bool | None:
    checkboxes = fields.get("checkboxes")
    if not isinstance(checkboxes, Mapping):
        return None
    payment_mode_keys = ["annual", "semi_annual", "quarterly", "monthly", "single_premium"]
    selected = sum(
        1
        for key in payment_mode_keys
        if normalize_checkbox_value(checkboxes.get(f"payment_mode.{key}")) is True
    )
    return selected <= 1


def _check_auto_debit_account(fields: Mapping[str, Any]) -> bool | None:
    checkboxes = fields.get("checkboxes")
    if not isinstance(checkboxes, Mapping):
        return None
    auto_debit_enabled = normalize_checkbox_value(checkboxes.get("payment_mode.auto_debit"))
    if auto_debit_enabled is not True:
        return None
    account_value = fields.get("auto_debit_account") or fields.get("bank_account_number")
    return isinstance(account_value, str) and bool(account_value.strip())
