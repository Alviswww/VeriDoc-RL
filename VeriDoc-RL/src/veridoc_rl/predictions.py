from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def parse_prediction_text(text: str, *, sample_id: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = _strip_json_fence(candidate)
    for payload_text in (fenced, candidate):
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return {
                "sample_id": payload.get("sample_id", sample_id),
                "fields": payload.get("fields", {}),
                "validations": payload.get("validations", []),
            }
    return {"sample_id": sample_id, "fields": {}, "validations": []}


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped
