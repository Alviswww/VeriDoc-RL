from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.evaluation import (
    compute_field_level_metrics,
    compute_form_exact_match,
    compute_invalid_json_rate,
    compute_rule_pass_rate,
    compute_validation_match_rate,
    evaluate_dataset,
    evaluate_prediction,
)
from veridoc_rl.evaluation.reporting import main as reporting_main


def _reference_record(
    sample_id: str = "sample-1",
    *,
    bucket: dict[str, str] | None = None,
) -> dict[str, object]:
    bucket = bucket or {
        "template_family": "template_a",
        "ocr_noise_level": "low",
        "hard_case_type": "field_missing",
        "rule_complexity": "single_field",
    }
    return {
        "input": {
            "sample_id": sample_id,
            "form_type": "insurance_application_form",
            "pdf_page": 1,
            "ocr_tokens": [],
        },
        "reference": {
            "sample_id": sample_id,
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
                    "rule_id": "格式.投保人联系电话",
                    "status": "pass",
                    "message": "投保人联系电话格式合法。",
                },
                {
                    "rule_id": "勾选.缴费方式互斥",
                    "status": "pass",
                    "message": "缴费方式勾选互斥。",
                },
            ],
        },
        "metadata": {"bucket": bucket},
    }


def test_field_metrics_handle_partial_match() -> None:
    prediction_fields = {"投保人姓名": "张三", "投保人联系电话": "13800138000"}
    reference_fields = {"投保人姓名": "张三", "投保人联系电话": "13700137000"}

    metrics = compute_field_level_metrics(prediction_fields, reference_fields)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_form_exact_match_normalizes_fields() -> None:
    prediction_fields = {"投保人联系电话": "138-0013-8000"}
    reference_fields = {"投保人联系电话": "13800138000"}

    assert compute_form_exact_match(prediction_fields, reference_fields) == 1.0


def test_rule_pass_rate_skips_not_applicable() -> None:
    validations = [
        {"status": "pass"},
        {"status": "fail"},
        {"status": "not_applicable"},
    ]

    assert compute_rule_pass_rate(validations) == 0.5


def test_invalid_json_rate_counts_schema_failures() -> None:
    items = [
        {"sample_id": "s1", "fields": {}, "validations": []},
        "{\"sample_id\": 1}",
        "not json",
    ]

    assert compute_invalid_json_rate(items) == 2 / 3


def test_validation_match_rate_detects_rule_status_mismatch() -> None:
    prediction_validations = [
        {"rule_id": "格式.投保人联系电话", "status": "fail"},
    ]
    reference_validations = [
        {"rule_id": "格式.投保人联系电话", "status": "pass"},
        {"rule_id": "勾选.缴费方式互斥", "status": "pass"},
    ]

    assert compute_validation_match_rate(prediction_validations, reference_validations) == 0.0


def test_evaluate_prediction_returns_core_metrics() -> None:
    prediction = {
        "sample_id": "sample-1",
        "fields": {"投保人联系电话": "138-0013-8000"},
        "validations": [
            {
                "rule_id": "格式.投保人联系电话",
                "status": "pass",
                "message": "投保人联系电话格式合法。",
            }
        ],
    }
    reference = {
        "sample_id": "sample-1",
        "fields": {"投保人联系电话": "13800138000"},
        "validations": [
            {
                "rule_id": "格式.投保人联系电话",
                "status": "pass",
                "message": "投保人联系电话格式合法。",
            }
        ],
    }

    metrics = evaluate_prediction(prediction=prediction, reference=reference)

    assert metrics["field_f1"] == 1.0
    assert metrics["form_exact_match"] == 1.0
    assert metrics["rule_pass_rate"] == 1.0
    assert metrics["validation_match_rate"] == 1.0
    assert metrics["invalid_json_rate"] == 0.0


def test_evaluate_dataset_summarizes_buckets_and_taxonomy() -> None:
    failure_reference = _reference_record(
        "sample-fail",
        bucket={
            "template_family": "template_c",
            "ocr_noise_level": "high",
            "hard_case_type": "checkbox_conflict",
            "rule_complexity": "conditional_checkbox",
        },
    )
    success_reference = _reference_record("sample-pass")

    failed_prediction = {
        "sample_id": "sample-fail",
        "fields": {
            "投保人姓名": "张三",
            "投保人联系电话": "",
            "投保人证件号码": "440101199001011234",
            "被保人姓名": "张三",
            "被保人证件号码": "440101199001011234",
            "被保人出生日期": "1990-01-01",
            "投被保人关系": "本人",
            "缴费方式": "年缴",
            "缴费年期": 20,
            "勾选项": {
                "缴费方式.年缴": True,
                "缴费方式.月缴": True,
                "缴费方式.自动扣款": False,
            },
        },
        "validations": [
            {
                "rule_id": "格式.投保人联系电话",
                "status": "fail",
                "message": "投保人联系电话缺失或格式不合法。",
            }
        ],
    }
    passed_prediction = success_reference["reference"]

    report = evaluate_dataset(
        [
            {
                "prediction": failed_prediction,
                "reference": failure_reference["reference"],
                "metadata": failure_reference["metadata"],
                "input": failure_reference["input"],
                "context": {"perturbed_predictions": [{"fields": {"投保人姓名": "李四"}}]},
            },
            {
                "prediction": passed_prediction,
                "reference": success_reference["reference"],
                "metadata": success_reference["metadata"],
                "input": success_reference["input"],
                "context": {
                    "perturbed_predictions": [
                        {"fields": dict(success_reference["reference"]["fields"])}
                    ]
                },
            },
        ],
        failure_case_limit=1,
    )

    assert report.overall["sample_count"] == 2
    assert report.overall["failure_count"] == 1
    assert report.overall["total_reward"] < 1.0
    assert report.bucket_metrics["ocr_noise_level"]["high"]["sample_count"] == 1
    assert report.error_taxonomy["missing_field"]["count"] == 1
    assert report.error_taxonomy["checkbox_logic_error"]["count"] == 1
    assert report.error_taxonomy["rule_misjudgment"]["count"] == 1
    assert report.error_taxonomy["ocr_noise_vulnerability"]["count"] == 1
    assert len(report.failure_cases) == 1
    assert report.failure_cases[0].sample_id == "sample-fail"
    assert "投保人联系电话" in report.failure_cases[0].missing_fields
    assert "missing_field" in report.failure_cases[0].taxonomy


def test_reporting_main_writes_report_and_case_export(tmp_path: Path) -> None:
    reference_record = _reference_record(
        "sample-cli",
        bucket={
            "template_family": "template_b",
            "ocr_noise_level": "medium",
            "hard_case_type": "field_missing",
            "rule_complexity": "single_field",
        },
    )
    prediction_record = {
        "prediction": {
            "sample_id": "sample-cli",
            "fields": {
                "投保人姓名": "张三",
                "投保人联系电话": "",
                "投保人证件号码": "440101199001011234",
                "被保人姓名": "张三",
                "被保人证件号码": "440101199001011234",
                "被保人出生日期": "1990-01-01",
                "投被保人关系": "本人",
                "缴费方式": "年缴",
                "缴费年期": 20,
                "勾选项": {
                    "缴费方式.年缴": True,
                    "缴费方式.月缴": True,
                    "缴费方式.自动扣款": False,
                },
            },
            "validations": [
                {
                    "rule_id": "格式.投保人联系电话",
                    "status": "fail",
                    "message": "投保人联系电话缺失或格式不合法。",
                }
            ],
        },
        "context": {"perturbed_predictions": [{"fields": {"投保人姓名": "李四"}}]},
    }
    reference_path = tmp_path / "reference.jsonl"
    prediction_path = tmp_path / "prediction.jsonl"
    report_path = tmp_path / "report.json"
    cases_path = tmp_path / "cases.jsonl"

    reference_path.write_text(
        json.dumps(reference_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(prediction_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = reporting_main(
        [
            "--reference-path",
            str(reference_path),
            "--prediction-path",
            str(prediction_path),
            "--report-path",
            str(report_path),
            "--case-export-path",
            str(cases_path),
            "--failure-only",
            "--failure-case-limit",
            "1",
        ]
    )

    assert exit_code == 0
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["overall"]["sample_count"] == 1
    assert report_payload["overall"]["total_reward"] < 1.0
    assert report_payload["bucket_metrics"]["template_family"]["template_b"]["sample_count"] == 1
    assert report_payload["error_taxonomy"]["missing_field"]["count"] == 1
    case_lines = cases_path.read_text(encoding="utf-8").splitlines()
    assert len(case_lines) == 1
    case_payload = json.loads(case_lines[0])
    assert case_payload["sample_id"] == "sample-cli"
    assert case_payload["reward"]["profile"] == "default"
    assert "missing_field" in case_payload["taxonomy"]
