from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast


ValidationStatus = Literal["pass", "fail", "not_applicable"]
VALIDATION_STATUSES: frozenset[str] = frozenset({"pass", "fail", "not_applicable"})


@dataclass(slots=True)
class OCRToken:
    text: str
    bbox: list[int]
    page: int

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("OCRToken.bbox must contain exactly four coordinates.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OCRToken":
        return cls(
            text=str(payload["text"]),
            bbox=[int(value) for value in payload["bbox"]],
            page=int(payload["page"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "bbox": list(self.bbox), "page": self.page}


@dataclass(slots=True)
class FormInput:
    sample_id: str
    form_type: str
    pdf_page: int
    ocr_tokens: list[OCRToken] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormInput":
        return cls(
            sample_id=str(payload["sample_id"]),
            form_type=str(payload["form_type"]),
            pdf_page=int(payload["pdf_page"]),
            ocr_tokens=[OCRToken.from_dict(item) for item in payload.get("ocr_tokens", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "form_type": self.form_type,
            "pdf_page": self.pdf_page,
            "ocr_tokens": [token.to_dict() for token in self.ocr_tokens],
        }


@dataclass(slots=True)
class ValidationResult:
    rule_id: str
    status: ValidationStatus
    message: str

    def __post_init__(self) -> None:
        if self.status not in VALIDATION_STATUSES:
            raise ValueError(f"Unsupported validation status: {self.status}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationResult":
        return cls(
            rule_id=str(payload["rule_id"]),
            status=cast(ValidationStatus, str(payload["status"])),
            message=str(payload["message"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {"rule_id": self.rule_id, "status": self.status, "message": self.message}


@dataclass(slots=True)
class FormOutput:
    sample_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    validations: list[ValidationResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormOutput":
        return cls(
            sample_id=str(payload["sample_id"]),
            fields=dict(payload.get("fields", {})),
            validations=[
                item if isinstance(item, ValidationResult) else ValidationResult.from_dict(item)
                for item in payload.get("validations", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "fields": dict(self.fields),
            "validations": [item.to_dict() for item in self.validations],
        }


def validate_prediction_payload(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    required_keys = {"sample_id", "fields", "validations"}
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        issues.append(f"missing keys: {', '.join(missing_keys)}")

    sample_id = payload.get("sample_id")
    if sample_id is not None and not isinstance(sample_id, str):
        issues.append("sample_id must be a string")

    fields = payload.get("fields")
    if fields is not None and not isinstance(fields, Mapping):
        issues.append("fields must be a mapping")

    validations = payload.get("validations")
    if validations is None:
        return issues
    if not isinstance(validations, list):
        issues.append("validations must be a list")
        return issues

    for index, item in enumerate(validations):
        if not isinstance(item, Mapping):
            issues.append(f"validations[{index}] must be a mapping")
            continue
        for key in ("rule_id", "status", "message"):
            if key not in item:
                issues.append(f"validations[{index}] missing {key}")
        status = item.get("status")
        if status is not None and status not in VALIDATION_STATUSES:
            issues.append(
                f"validations[{index}].status must be one of {sorted(VALIDATION_STATUSES)}"
            )

    return issues
