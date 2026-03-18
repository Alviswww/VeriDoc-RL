from __future__ import annotations

from veridoc_rl.verifiers import (
    CheckboxLogicVerifier,
    CrossFieldConsistencyVerifier,
    FieldMatchVerifier,
    NormalizationVerifier,
    OCRRobustnessVerifier,
    SchemaVerifier,
    build_default_verifiers,
    run_verifier_suite,
)


def _reference_payload() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "fields": {
            "policyholder_name": "张三",
            "policyholder_phone": "13800138000",
            "policyholder_id_number": "440101199001011234",
            "insured_name": "张三",
            "insured_id_number": "440101199001011234",
            "insured_birth_date": "1990-01-01",
            "relation_policyholder_to_insured": "self",
            "payment_mode": "annual",
            "payment_period_years": 20,
            "checkboxes": {
                "payment_mode.annual": True,
                "payment_mode.monthly": False,
                "payment_mode.auto_debit": False,
            },
        },
        "validations": [
            {
                "rule_id": "required.policyholder_name",
                "status": "pass",
                "message": "policyholder_name is present",
            }
        ],
    }


def test_schema_verifier_accepts_valid_payload() -> None:
    result = SchemaVerifier().verify(_reference_payload())

    assert result.passed is True
    assert result.score == 1.0


def test_field_match_verifier_uses_normalization() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "policyholder_phone": "138-0013-8000",
    }

    result = FieldMatchVerifier().verify(prediction=prediction, reference=_reference_payload())

    assert result.passed is True
    assert result.score == 1.0


def test_normalization_verifier_flags_non_canonical_values() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "policyholder_phone": "138-0013-8000",
    }

    result = NormalizationVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert result.score < 1.0
    assert "policyholder_phone" in result.details["invalid_fields"]


def test_cross_field_consistency_verifier_detects_birth_date_mismatch() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "insured_birth_date": "1991-01-01",
    }

    result = CrossFieldConsistencyVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert "birth_date_vs_id_number" in result.details["failed_checks"]


def test_checkbox_logic_verifier_detects_conflicting_selection() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "checkboxes": {
            "payment_mode.annual": True,
            "payment_mode.monthly": True,
            "payment_mode.auto_debit": False,
        },
    }

    result = CheckboxLogicVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert "payment_mode_exclusive" in result.details["failed_checks"]


def test_ocr_robustness_verifier_scores_perturbed_predictions() -> None:
    prediction = _reference_payload()
    context = {"perturbed_predictions": [{"fields": dict(prediction["fields"])}]}

    result = OCRRobustnessVerifier().verify(prediction=prediction, context=context)

    assert result.passed is True
    assert result.score == 1.0


def test_run_verifier_suite_uses_default_phase_a_order() -> None:
    prediction = _reference_payload()
    results = run_verifier_suite(prediction=prediction, reference=_reference_payload())

    assert [result.name for result in results] == [
        verifier.name for verifier in build_default_verifiers()
    ]
    assert len(results) == 6
    assert all(result.passed for result in results)
