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
            "投保人姓名": "张三",
            "投保人联系电话": "13800138000",
            "投保人证件号码": "440101199001011234",
            "被保人姓名": "张三",
            "被保人证件号码": "440101199001011234",
            "被保人出生日期": "1990-01-01",
            "投被保人关系": "本人",
            "缴费方式": "年缴",
            "缴费年期": 20,
            "勾选项": {
                "缴费方式.年缴": True,
                "缴费方式.月缴": False,
                "缴费方式.自动扣款": False,
            },
        },
        "validations": [
            {
                "rule_id": "必填.投保人姓名",
                "status": "pass",
                "message": "投保人姓名已填写。",
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
        "投保人联系电话": "138-0013-8000",
    }

    result = FieldMatchVerifier().verify(prediction=prediction, reference=_reference_payload())

    assert result.passed is True
    assert result.score == 1.0


def test_normalization_verifier_flags_non_canonical_values() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "投保人联系电话": "138-0013-8000",
    }

    result = NormalizationVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert result.score < 1.0
    assert "投保人联系电话" in result.details["invalid_fields"]


def test_cross_field_consistency_verifier_detects_birth_date_mismatch() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "被保人出生日期": "1991-01-01",
    }

    result = CrossFieldConsistencyVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert "出生日期与证件号码一致" in result.details["failed_checks"]


def test_checkbox_logic_verifier_detects_conflicting_selection() -> None:
    prediction = _reference_payload()
    prediction["fields"] = {
        **prediction["fields"],
        "勾选项": {
            "缴费方式.年缴": True,
            "缴费方式.月缴": True,
            "缴费方式.自动扣款": False,
        },
    }

    result = CheckboxLogicVerifier().verify(prediction=prediction)

    assert result.passed is False
    assert "缴费方式互斥" in result.details["failed_checks"]


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
