from __future__ import annotations

import json
from pathlib import Path

import veridoc_rl.inference.candidates as candidates_module
from veridoc_rl.inference.candidates import CandidateGenerationConfig, generate_candidates_for_records
from veridoc_rl.model_defaults import DEFAULT_BASELINE_MODEL


def _input_record() -> dict[str, object]:
    return {
        "input": {
            "sample_id": "sample-1",
            "form_type": "insurance_application_form",
            "pdf_page": 1,
            "ocr_tokens": [{"text": "投保人姓名", "bbox": [1, 2, 3, 4], "page": 1}],
        },
        "metadata": {"bucket": {"template_family": "template_a"}},
    }


def test_generate_candidates_for_records_parses_json_choices(monkeypatch) -> None:
    monkeypatch.setattr(
        candidates_module,
        "_post_json",
        lambda **_: {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"sample_id":"sample-1","fields":{"policyholder_name":"张三"},"validations":[]}\n'
                            "```"
                        )
                    }
                },
                {
                    "message": {
                        "content": (
                            '{"sample_id":"sample-1","fields":{"policyholder_name":"李四"},"validations":[]}'
                        )
                    }
                },
            ]
        },
    )
    config = CandidateGenerationConfig(
        model=DEFAULT_BASELINE_MODEL,
        backend="sglang",
        num_candidates=2,
    )

    rows = generate_candidates_for_records([_input_record()], config=config)

    assert len(rows) == 2
    assert rows[0]["candidate_id"] == "sample-1::cand_0"
    assert rows[0]["prediction"]["fields"]["policyholder_name"] == "张三"
    assert rows[1]["prediction"]["fields"]["policyholder_name"] == "李四"
    assert rows[0]["backend"] == "sglang"


def test_candidate_cli_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "candidates.jsonl"
    input_path.write_text(json.dumps(_input_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        candidates_module,
        "_post_json",
        lambda **_: {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"sample_id":"sample-1","fields":{"policyholder_name":"张三"},"validations":[]}'
                        )
                    }
                }
            ]
        },
    )

    exit_code = candidates_module.main(
        [
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--model",
            DEFAULT_BASELINE_MODEL,
            "--num-candidates",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["model"] == DEFAULT_BASELINE_MODEL
    assert payload["prediction"]["sample_id"] == "sample-1"


def test_request_chat_candidates_rejects_unknown_backend() -> None:
    config = CandidateGenerationConfig(model=DEFAULT_BASELINE_MODEL, backend="unknown")

    try:
        candidates_module.request_chat_candidates(input_payload=_input_record()["input"], config=config)
    except ValueError as exc:
        assert "Unsupported inference backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported backend.")


def test_candidate_cli_expands_env_backed_model_and_api_base(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "candidates.jsonl"
    input_path.write_text(json.dumps(_input_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("VERIDOC_MODEL_REF", str(tmp_path / "models" / "Qwen3-1.7B"))
    monkeypatch.setenv("VERIDOC_API_BASE", "http://127.0.0.1:30000/v1")

    seen: dict[str, str] = {}

    def _fake_post_json(**kwargs):  # type: ignore[no-untyped-def]
        seen["url"] = kwargs["url"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"sample_id":"sample-1","fields":{"policyholder_name":"张三"},"validations":[]}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(candidates_module, "_post_json", _fake_post_json)

    exit_code = candidates_module.main(
        [
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--model",
            "${VERIDOC_MODEL_REF}",
            "--api-base",
            "${VERIDOC_API_BASE}",
            "--num-candidates",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["model"] == str(tmp_path / "models" / "Qwen3-1.7B")
    assert seen["url"] == "http://127.0.0.1:30000/v1/chat/completions"
