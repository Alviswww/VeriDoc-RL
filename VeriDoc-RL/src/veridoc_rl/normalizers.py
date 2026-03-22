from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from veridoc_rl.form_spec import (
    FIELD_APPLICATION_DATE,
    FIELD_BENEFICIARY_RATIO,
    FIELD_COVERAGE_AMOUNT,
    FIELD_CURRENCY,
    FIELD_INSURED_BIRTH_DATE,
    FIELD_PAYMENT_MODE,
    FIELD_POLICYHOLDER_GENDER,
    FIELD_POLICYHOLDER_ID_NUMBER,
    FIELD_POLICYHOLDER_PHONE,
    FIELD_RELATION,
    GENDER_ALIASES,
    PAYMENT_MODE_ALIASES,
    PRODUCT_NAME_ALIASES,
    RELATION_ALIASES,
    canonicalize_field_name,
)


_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")
_TRUE_VALUES = {"1", "true", "yes", "y", "checked", "selected", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "unchecked", "off"}


def normalize_phone(phone: Any) -> str | None:
    if phone is None:
        return None
    digits = re.sub(r"\D", "", str(phone))
    if digits.startswith("86") and len(digits) == 13:
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("1"):
        return digits
    return None


def normalize_id_number(id_number: Any) -> str | None:
    if id_number is None:
        return None
    normalized = re.sub(r"\s+", "", str(id_number)).upper()
    if re.fullmatch(r"\d{17}[\dX]", normalized):
        return normalized
    return None


def extract_birth_date_from_id_number(id_number: str) -> str | None:
    normalized = normalize_id_number(id_number)
    if normalized is None:
        return None
    return normalize_date(normalized[6:14])


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    chinese_match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", raw)
    if chinese_match:
        year, month, day = (int(part) for part in chinese_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_amount(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        decimal_value = Decimal(str(value))
    else:
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace(",", "")
        raw = raw.replace("￥", "").replace("¥", "")
        try:
            decimal_value = Decimal(raw)
        except InvalidOperation:
            return None

    if decimal_value == decimal_value.to_integral():
        return str(decimal_value.quantize(Decimal("1")))
    return format(decimal_value.normalize(), "f")


def normalize_checkbox_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return None


def normalize_known_field(field_name: str, value: Any) -> Any | None:
    if value is None:
        return None

    canonical_name = canonicalize_field_name(field_name)
    if canonical_name == FIELD_POLICYHOLDER_PHONE:
        return normalize_phone(value)
    if canonical_name == FIELD_POLICYHOLDER_ID_NUMBER or canonical_name.endswith("证件号码"):
        return normalize_id_number(value)
    if canonical_name in {FIELD_APPLICATION_DATE, FIELD_INSURED_BIRTH_DATE} or canonical_name.endswith("日期"):
        return normalize_date(value)
    if canonical_name == FIELD_COVERAGE_AMOUNT or "金额" in canonical_name or "保额" in canonical_name:
        return normalize_amount(value)
    if canonical_name == FIELD_PAYMENT_MODE and isinstance(value, str):
        normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
        return PAYMENT_MODE_ALIASES.get(normalized) or PAYMENT_MODE_ALIASES.get(value.strip()) or value.strip()
    if canonical_name == FIELD_CURRENCY and isinstance(value, str):
        return value.strip().upper()
    if canonical_name == FIELD_RELATION and isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        return RELATION_ALIASES.get(normalized) or RELATION_ALIASES.get(value.strip()) or value.strip()
    if canonical_name in {FIELD_POLICYHOLDER_GENDER, "被保人性别"} and isinstance(value, str):
        normalized = value.strip().lower()
        return GENDER_ALIASES.get(normalized) or GENDER_ALIASES.get(value.strip()) or value.strip()
    if canonical_name == "产品名称" and isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        return PRODUCT_NAME_ALIASES.get(normalized) or PRODUCT_NAME_ALIASES.get(value.strip()) or value.strip()
    if canonical_name == FIELD_BENEFICIARY_RATIO:
        if isinstance(value, (int, float)):
            return str(int(value)) if float(value).is_integer() else str(value)
        if isinstance(value, str):
            normalized = value.strip().rstrip("%")
            return normalized or None
    return None
