from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "你是保险投保单结构化抽取与规则校验助手。"
    "请仅输出 JSON，对 OCR 结果做字段抽取并返回 validations。"
)


def build_user_prompt(input_payload: Mapping[str, Any]) -> str:
    sample_id = str(input_payload.get("sample_id", ""))
    form_type = str(input_payload.get("form_type", "insurance_application_form"))
    pdf_page = input_payload.get("pdf_page", 1)
    ocr_tokens = _format_ocr_tokens(input_payload.get("ocr_tokens"))
    schema_hint = (
        '输出 JSON schema: {"sample_id": "...", "fields": {...}, '
        '"validations": [{"rule_id": "...", "status": "pass|fail|not_applicable", '
        '"message": "..."}]}.'
    )
    return "\n".join(
        [
            "任务: 根据 OCR token 结果提取投保单字段，并同时输出规则校验结果。",
            f"sample_id: {sample_id}",
            f"form_type: {form_type}",
            f"pdf_page: {pdf_page}",
            schema_hint,
            "OCR tokens:",
            ocr_tokens,
        ]
    )


def build_assistant_response(reference_payload: Mapping[str, Any]) -> str:
    serializable = {
        "sample_id": reference_payload.get("sample_id"),
        "fields": reference_payload.get("fields", {}),
        "validations": reference_payload.get("validations", []),
    }
    return json.dumps(serializable, ensure_ascii=False, indent=2)


def build_chat_messages(
    input_payload: Mapping[str, Any],
    *,
    reference_payload: Mapping[str, Any] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(input_payload)},
    ]
    if reference_payload is not None:
        messages.append(
            {
                "role": "assistant",
                "content": build_assistant_response(reference_payload),
            }
        )
    return messages


def _format_ocr_tokens(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "[]"

    rendered_lines: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        rendered_lines.append(
            json.dumps(
                {
                    "index": index,
                    "text": item.get("text", ""),
                    "bbox": item.get("bbox", []),
                    "page": item.get("page", 1),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rendered_lines) if rendered_lines else "[]"
