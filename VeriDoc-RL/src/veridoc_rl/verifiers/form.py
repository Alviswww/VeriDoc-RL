from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from veridoc_rl.form_spec import (
    CHECKBOX_AUTO_DEBIT,
    CHECKBOX_PAYMENT_ANNUAL,
    CHECKBOX_PAYMENT_MONTHLY,
    CHECKBOX_PAYMENT_QUARTERLY,
    CHECKBOX_PAYMENT_SEMI_ANNUAL,
    CHECKBOX_PAYMENT_SINGLE,
    FIELD_APPLICATION_DATE,
    FIELD_AUTO_DEBIT_ACCOUNT,
    FIELD_BENEFICIARY_RATIO,
    FIELD_CHECKBOXES,
    FIELD_INSURED_BIRTH_DATE,
    FIELD_INSURED_ID_NUMBER,
    FIELD_PAYMENT_MODE,
    FIELD_PAYMENT_PERIOD_YEARS,
    FIELD_POLICYHOLDER_ID_NUMBER,
    FIELD_PRODUCT_NAME,
    FIELD_RELATION,
    PAYMENT_MODE_ANNUAL,
    PAYMENT_MODE_MONTHLY,
    PAYMENT_MODE_QUARTERLY,
    PAYMENT_MODE_SEMI_ANNUAL,
    PAYMENT_MODE_SINGLE,
    RELATION_CHILD,
    RELATION_EMPLOYEE,
    RELATION_OTHER,
    RELATION_PARENT,
    RELATION_SELF,
    RELATION_SPOUSE,
    canonicalize_fields,
)
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

        prediction_fields = _extract_fields(prediction)
        reference_fields = _extract_fields(reference)
        if prediction_fields is None or reference_fields is None:
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
            if _canonicalize_field_value(field_name, actual_value) == _canonicalize_field_value(field_name, expected_value):
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
        prediction_fields = _extract_fields(prediction)
        if prediction_fields is None:
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
        prediction_fields = _extract_fields(prediction)
        if prediction_fields is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        checks: dict[str, bool | None] = {
            "出生日期与证件号码一致": _check_birth_date_vs_id(prediction_fields),
            "受益比例合计": _check_beneficiary_ratio_sum(prediction_fields),
            "投被保人关系": _check_relation(prediction_fields, context),
            "产品缴费组合": _check_product_payment_combo(prediction_fields, context),
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
        prediction_fields = _extract_fields(prediction)
        if prediction_fields is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                name=self.name,
                details={"reason": "fields_missing_or_invalid"},
            )

        checks: dict[str, bool | None] = {
            "缴费方式互斥": _check_payment_mode_exclusive(prediction_fields),
            "自动扣款需账户": _check_auto_debit_account(prediction_fields),
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
        prediction_fields = _extract_fields(prediction)
        if prediction_fields is None:
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
            candidate_fields = _extract_fields(item) if isinstance(item, Mapping) else None
            if candidate_fields is None:
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


def _extract_fields(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        return None
    return canonicalize_fields(fields)


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
    birth_date = fields.get(FIELD_INSURED_BIRTH_DATE)
    id_number = fields.get(FIELD_INSURED_ID_NUMBER) or fields.get(FIELD_POLICYHOLDER_ID_NUMBER)
    normalized_id = normalize_id_number(id_number)
    normalized_birth = normalize_known_field(FIELD_INSURED_BIRTH_DATE, birth_date)
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
    direct_ratio = _parse_ratio(fields.get(FIELD_BENEFICIARY_RATIO))
    if direct_ratio is not None:
        ratios.append(direct_ratio)
    return ratios


def _check_beneficiary_ratio_sum(fields: Mapping[str, Any]) -> bool | None:
    ratios = _extract_beneficiary_ratios(fields)
    if not ratios:
        return None
    return abs(sum(ratios) - 100.0) < 1e-6


def _check_relation(fields: Mapping[str, Any], context: Mapping[str, Any] | None) -> bool | None:
    relation = normalize_known_field(FIELD_RELATION, fields.get(FIELD_RELATION))
    if relation is None:
        return None
    default_relations = [
        RELATION_SELF,
        RELATION_SPOUSE,
        RELATION_CHILD,
        RELATION_PARENT,
        RELATION_EMPLOYEE,
        RELATION_OTHER,
    ]
    allowed_relations = {
        normalize_known_field(FIELD_RELATION, item)
        for item in (context or {}).get("allowed_relations", default_relations)
    }
    return relation in allowed_relations


def _check_product_payment_combo(
    fields: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> bool | None:
    product_name = fields.get(FIELD_PRODUCT_NAME)
    payment_mode = normalize_known_field(FIELD_PAYMENT_MODE, fields.get(FIELD_PAYMENT_MODE))
    payment_period = fields.get(FIELD_PAYMENT_PERIOD_YEARS)
    if product_name is None and payment_mode is None and payment_period is None:
        return None
    default_modes = [
        PAYMENT_MODE_ANNUAL,
        PAYMENT_MODE_SEMI_ANNUAL,
        PAYMENT_MODE_QUARTERLY,
        PAYMENT_MODE_MONTHLY,
        PAYMENT_MODE_SINGLE,
    ]
    allowed_modes = {
        normalize_known_field(FIELD_PAYMENT_MODE, item)
        for item in (context or {}).get("allowed_payment_modes", default_modes)
    }
    if payment_mode is not None and payment_mode not in allowed_modes:
        return False
    if payment_mode == PAYMENT_MODE_SINGLE:
        return payment_period in (None, 1, "1")
    return payment_period is None or str(payment_period).isdigit()


def _check_payment_mode_exclusive(fields: Mapping[str, Any]) -> bool | None:
    checkboxes = fields.get(FIELD_CHECKBOXES)
    if not isinstance(checkboxes, Mapping):
        return None
    payment_mode_keys = [
        CHECKBOX_PAYMENT_ANNUAL,
        CHECKBOX_PAYMENT_SEMI_ANNUAL,
        CHECKBOX_PAYMENT_QUARTERLY,
        CHECKBOX_PAYMENT_MONTHLY,
        CHECKBOX_PAYMENT_SINGLE,
    ]
    selected = sum(1 for key in payment_mode_keys if normalize_checkbox_value(checkboxes.get(key)) is True)
    return selected <= 1


def _check_auto_debit_account(fields: Mapping[str, Any]) -> bool | None:
    checkboxes = fields.get(FIELD_CHECKBOXES)
    if not isinstance(checkboxes, Mapping):
        return None
    auto_debit_enabled = normalize_checkbox_value(checkboxes.get(CHECKBOX_AUTO_DEBIT))
    if auto_debit_enabled is not True:
        return None
    account_value = fields.get(FIELD_AUTO_DEBIT_ACCOUNT)
    return isinstance(account_value, str) and bool(account_value.strip())
