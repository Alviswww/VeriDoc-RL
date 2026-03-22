from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.data.preferences import build_preference_pairs, main


def _reference_record() -> dict[str, object]:
    return {
        "input": {
            "sample_id": "sample-pref",
            "form_type": "insurance_application_form",
            "pdf_page": 1,
            "ocr_tokens": [],
        },
        "reference": {
            "sample_id": "sample-pref",
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
        },
        "metadata": {"bucket": {"template_family": "template_a"}},
    }


def test_build_preference_pairs_selects_best_candidate() -> None:
    reference_records = [_reference_record()]
    candidate_records = [
        {
            "candidate_id": "good",
            "prediction": reference_records[0]["reference"],
        },
        {
            "candidate_id": "bad",
            "prediction": {
                "sample_id": "sample-pref",
                "fields": {
                    "投保人姓名": "张三",
                    "投保人联系电话": "",
                    "投保人证件号码": "440101199001011234",
                    "被保人姓名": "张三",
                    "被保人证件号码": "440101199001011234",
                    "被保人出生日期": "1991-01-01",
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
                        "rule_id": "必填.投保人姓名",
                        "status": "pass",
                        "message": "投保人姓名已填写。",
                    }
                ],
            },
        },
    ]

    examples = build_preference_pairs(candidate_records, reference_records, min_margin=0.05)

    assert len(examples) == 1
    assert examples[0].chosen_candidate_id == "good"
    assert examples[0].rejected_candidate_id == "bad"
    assert examples[0].reward_margin > 0


def test_preference_cli_writes_jsonl(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output_path = tmp_path / "preferences.jsonl"

    reference_path.write_text(
        json.dumps(_reference_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "good",
                        "prediction": _reference_record()["reference"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "candidate_id": "bad",
                        "prediction": {
                            "sample_id": "sample-pref",
                            "fields": {"投保人姓名": ""},
                            "validations": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--reference-path",
            str(reference_path),
            "--candidate-path",
            str(candidate_path),
            "--output-path",
            str(output_path),
            "--min-margin",
            "0.05",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["task_type"] == "DPO_preference"
    assert payload["chosen_candidate_id"] == "good"
    assert payload["reward_profile"] == "default"
