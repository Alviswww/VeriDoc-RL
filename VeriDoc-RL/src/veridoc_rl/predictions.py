from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from veridoc_rl.form_spec import canonicalize_prediction_payload


def parse_prediction_text(text: str, *, sample_id: str) -> dict[str, Any]:
    candidate = _strip_thinking_blocks(text.strip())
    candidates = _candidate_payload_texts(candidate)
    for payload_text in candidates:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return canonicalize_prediction_payload(payload, sample_id=sample_id)
    return {"sample_id": sample_id, "fields": {}, "validations": []}


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _candidate_payload_texts(text: str) -> list[str]:
    stripped = _strip_json_fence(text)
    candidates = [stripped, text]
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match is not None:
        candidates.insert(0, object_match.group(0))
    return list(dict.fromkeys(item.strip() for item in candidates if item.strip()))
