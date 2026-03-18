from __future__ import annotations

import pytest

from veridoc_rl.schema import FormInput, FormOutput, OCRToken, ValidationResult, validate_prediction_payload


def test_form_input_round_trip() -> None:
    payload = {
        "sample_id": "sample-1",
        "form_type": "insurance_application_form",
        "pdf_page": 1,
        "ocr_tokens": [{"text": "姓名", "bbox": [1, 2, 3, 4], "page": 1}],
    }

    form_input = FormInput.from_dict(payload)

    assert form_input.to_dict() == payload


def test_form_output_round_trip() -> None:
    output = FormOutput(
        sample_id="sample-1",
        fields={"policyholder_name": "张三"},
        validations=[
            ValidationResult(
                rule_id="required.policyholder_name",
                status="pass",
                message="policyholder_name is present",
            )
        ],
    )

    assert FormOutput.from_dict(output.to_dict()) == output


def test_validation_payload_reports_issues() -> None:
    issues = validate_prediction_payload({"sample_id": 123, "fields": [], "validations": [{}]})

    assert "sample_id must be a string" in issues
    assert "fields must be a mapping" in issues
    assert "validations[0] missing rule_id" in issues


def test_validation_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        ValidationResult(rule_id="rule", status="unknown", message="bad status")


def test_ocr_token_requires_four_coordinates() -> None:
    with pytest.raises(ValueError):
        OCRToken(text="姓名", bbox=[1, 2, 3], page=1)
