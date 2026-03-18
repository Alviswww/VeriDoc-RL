from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from veridoc_rl.normalizers import (
    extract_birth_date_from_id_number,
    normalize_checkbox_value,
    normalize_date,
    normalize_id_number,
    normalize_known_field,
    normalize_phone,
)
from veridoc_rl.schema import FormInput, FormOutput, OCRToken, ValidationResult


TEMPLATE_FAMILIES: tuple[str, ...] = ("template_a", "template_b", "template_c")
OCR_NOISE_LEVELS: tuple[str, ...] = ("low", "medium", "high")
HARD_CASE_TYPES: tuple[str, ...] = (
    "field_missing",
    "smudged_text",
    "occluded_region",
    "checkbox_conflict",
)
RULE_COMPLEXITIES: tuple[str, ...] = (
    "single_field",
    "cross_field",
    "conditional_checkbox",
)
TASK_TYPES: tuple[str, ...] = ("SFT_gold", "SFT_silver", "RL_prompt_only")

_TEMPLATE_BASE_X = {"template_a": 16, "template_b": 28, "template_c": 40}
_TEMPLATE_BASE_Y = {"template_a": 24, "template_b": 32, "template_c": 40}
_NAME_POOL = ("张三", "李四", "王敏", "陈静", "赵磊", "刘洋")
_ADDRESS_POOL = (
    "广东省广州市天河区体育东路138号",
    "上海市浦东新区张江路88号",
    "浙江省杭州市西湖区文三路99号",
)
_PRODUCTS = ("whole_life", "critical_illness", "medical")
_RELATIONS = ("self", "spouse", "child", "parent")
_PAYMENT_MODES = ("annual", "monthly", "single_premium")
_CURRENCY = ("CNY", "USD")

_FIELD_LAYOUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "template_a": (
        ("policyholder_name", "投保人姓名"),
        ("policyholder_phone", "联系电话"),
        ("policyholder_id_number", "证件号码"),
        ("insured_name", "被保人姓名"),
        ("insured_birth_date", "出生日期"),
        ("product_name", "险种名称"),
        ("coverage_amount", "保额"),
        ("payment_mode", "缴费方式"),
        ("application_date", "申请日期"),
    ),
    "template_b": (
        ("policyholder_name", "投保人"),
        ("policyholder_phone", "手机号"),
        ("policyholder_id_number", "身份证号"),
        ("insured_name", "被保险人"),
        ("insured_birth_date", "生日"),
        ("product_name", "产品"),
        ("coverage_amount", "保险金额"),
        ("payment_mode", "支付方式"),
        ("application_date", "投保日期"),
    ),
    "template_c": (
        ("policyholder_name", "姓名(投保人)"),
        ("policyholder_phone", "手机"),
        ("policyholder_id_number", "证件号"),
        ("insured_name", "姓名(被保人)"),
        ("insured_birth_date", "出生年月日"),
        ("product_name", "产品计划"),
        ("coverage_amount", "保障额度"),
        ("payment_mode", "缴费周期"),
        ("application_date", "填写日期"),
    ),
}


@dataclass(slots=True)
class SyntheticSample:
    input: FormInput
    reference: FormOutput
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "reference": self.reference.to_dict(),
            "metadata": dict(self.metadata),
        }


class SyntheticFormGenerator:
    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def generate_sample(
        self,
        sample_index: int,
        template_family: str | None = None,
        ocr_noise_level: str | None = None,
        hard_case_type: str | None = None,
        rule_complexity: str | None = None,
    ) -> SyntheticSample:
        template_family = template_family or TEMPLATE_FAMILIES[sample_index % len(TEMPLATE_FAMILIES)]
        ocr_noise_level = ocr_noise_level or OCR_NOISE_LEVELS[sample_index % len(OCR_NOISE_LEVELS)]
        hard_case_type = hard_case_type or HARD_CASE_TYPES[sample_index % len(HARD_CASE_TYPES)]
        rule_complexity = rule_complexity or RULE_COMPLEXITIES[sample_index % len(RULE_COMPLEXITIES)]

        fields = self._build_fields(sample_index=sample_index, hard_case_type=hard_case_type)
        ocr_tokens = self._build_ocr_tokens(
            fields=fields,
            template_family=template_family,
            ocr_noise_level=ocr_noise_level,
            hard_case_type=hard_case_type,
        )
        sample_id = f"{template_family}_{sample_index:05d}"
        form_input = FormInput(
            sample_id=sample_id,
            form_type="insurance_application_form",
            pdf_page=1,
            ocr_tokens=ocr_tokens,
        )
        validations = build_validations(fields)
        form_output = FormOutput(sample_id=sample_id, fields=fields, validations=validations)
        metadata = {
            "bucket": {
                "template_family": template_family,
                "ocr_noise_level": ocr_noise_level,
                "hard_case_type": hard_case_type,
                "rule_complexity": rule_complexity,
            },
            "generator_seed": self._seed,
            "sample_index": sample_index,
        }
        return SyntheticSample(input=form_input, reference=form_output, metadata=metadata)

    def generate_dataset(
        self,
        count: int,
        template_families: Sequence[str] | None = None,
        ocr_noise_levels: Sequence[str] | None = None,
        hard_case_types: Sequence[str] | None = None,
        rule_complexities: Sequence[str] | None = None,
        task_type: str = "SFT_gold",
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        if task_type not in TASK_TYPES:
            supported = ", ".join(TASK_TYPES)
            raise ValueError(f"Unsupported task_type: {task_type}. Expected one of: {supported}")

        templates = tuple(template_families or TEMPLATE_FAMILIES)
        noise_levels = tuple(ocr_noise_levels or OCR_NOISE_LEVELS)
        hard_cases = tuple(hard_case_types or HARD_CASE_TYPES)
        complexities = tuple(rule_complexities or RULE_COMPLEXITIES)

        records: list[dict[str, Any]] = []
        for index in range(count):
            sample = self.generate_sample(
                sample_index=index,
                template_family=templates[index % len(templates)],
                ocr_noise_level=noise_levels[index % len(noise_levels)],
                hard_case_type=hard_cases[index % len(hard_cases)],
                rule_complexity=complexities[index % len(complexities)],
            )
            records.append(build_training_record(sample, task_type=task_type))
        return records

    def _build_fields(self, sample_index: int, hard_case_type: str) -> dict[str, Any]:
        insured_name = self._rng.choice(_NAME_POOL)
        policyholder_name = insured_name if sample_index % 2 == 0 else self._rng.choice(_NAME_POOL)
        birth_date = f"19{90 + sample_index % 10:02d}-{(sample_index % 12) + 1:02d}-{(sample_index % 27) + 1:02d}"
        id_number = self._build_id_number(birth_date=birth_date, sequence=120 + sample_index)
        payment_mode = _PAYMENT_MODES[sample_index % len(_PAYMENT_MODES)]
        relation = "self" if policyholder_name == insured_name else _RELATIONS[(sample_index % (len(_RELATIONS) - 1)) + 1]
        fields: dict[str, Any] = {
            "policyholder_name": policyholder_name,
            "policyholder_gender": "male" if sample_index % 2 == 0 else "female",
            "policyholder_id_number": id_number,
            "policyholder_phone": f"1380013{sample_index % 10000:04d}",
            "policyholder_address": _ADDRESS_POOL[sample_index % len(_ADDRESS_POOL)],
            "insured_name": insured_name,
            "insured_gender": "male" if sample_index % 2 == 0 else "female",
            "insured_id_number": id_number,
            "insured_birth_date": birth_date,
            "relation_policyholder_to_insured": relation,
            "product_name": _PRODUCTS[sample_index % len(_PRODUCTS)],
            "coverage_amount": str(100000 * ((sample_index % 5) + 1)),
            "currency": _CURRENCY[sample_index % len(_CURRENCY)],
            "payment_mode": payment_mode,
            "payment_period_years": 1 if payment_mode == "single_premium" else 20,
            "beneficiary_name": self._rng.choice(_NAME_POOL),
            "beneficiary_ratio": "100",
            "signature_present": True,
            "application_date": f"2024-{((sample_index + 1) % 12) + 1:02d}-{((sample_index + 9) % 27) + 1:02d}",
            "checkboxes": {
                "payment_mode.annual": payment_mode == "annual",
                "payment_mode.monthly": payment_mode == "monthly",
                "payment_mode.single_premium": payment_mode == "single_premium",
                "payment_mode.auto_debit": sample_index % 3 == 0,
            },
        }

        if fields["checkboxes"]["payment_mode.auto_debit"]:
            fields["auto_debit_account"] = f"622202{sample_index % 10000000000:010d}"

        if hard_case_type == "field_missing":
            fields.pop("policyholder_phone", None)
        elif hard_case_type == "checkbox_conflict":
            fields["checkboxes"]["payment_mode.annual"] = True
            fields["checkboxes"]["payment_mode.monthly"] = True
        return fields

    def _build_ocr_tokens(
        self,
        fields: dict[str, Any],
        template_family: str,
        ocr_noise_level: str,
        hard_case_type: str,
    ) -> list[OCRToken]:
        tokens: list[OCRToken] = []
        base_x = _TEMPLATE_BASE_X[template_family]
        base_y = _TEMPLATE_BASE_Y[template_family]
        for index, (field_name, label) in enumerate(_FIELD_LAYOUTS[template_family]):
            y = base_y + index * 34
            tokens.append(OCRToken(text=label, bbox=[base_x, y, base_x + 90, y + 18], page=1))
            if field_name not in fields:
                continue
            value_text = self._serialize_value(field_name, fields[field_name])
            if hard_case_type == "occluded_region" and field_name in {"coverage_amount", "application_date"}:
                continue
            if hard_case_type == "smudged_text" and field_name in {"policyholder_name", "product_name"}:
                value_text = apply_ocr_noise(value_text, "high", self._rng)
            else:
                value_text = apply_ocr_noise(value_text, ocr_noise_level, self._rng)
            tokens.append(
                OCRToken(
                    text=value_text,
                    bbox=[base_x + 110, y, base_x + 260, y + 18],
                    page=1,
                )
            )

        checkbox_text = self._serialize_checkboxes(fields.get("checkboxes", {}))
        tokens.append(
            OCRToken(
                text=apply_ocr_noise(checkbox_text, ocr_noise_level, self._rng),
                bbox=[base_x, base_y + 340, base_x + 280, base_y + 360],
                page=1,
            )
        )
        return tokens

    def _serialize_value(self, field_name: str, value: Any) -> str:
        if field_name == "coverage_amount":
            return f"{value}元"
        if field_name == "application_date":
            normalized = normalize_date(value) or str(value)
            return normalized.replace("-", "/")
        return str(value)

    def _serialize_checkboxes(self, checkboxes: Any) -> str:
        if not isinstance(checkboxes, dict):
            return ""
        annual = "√" if checkboxes.get("payment_mode.annual") else "□"
        monthly = "√" if checkboxes.get("payment_mode.monthly") else "□"
        single = "√" if checkboxes.get("payment_mode.single_premium") else "□"
        auto_debit = "√" if checkboxes.get("payment_mode.auto_debit") else "□"
        return f"年缴{annual} 月缴{monthly} 趸缴{single} 自动扣款{auto_debit}"

    def _build_id_number(self, birth_date: str, sequence: int) -> str:
        normalized_birth = normalize_date(birth_date)
        if normalized_birth is None:
            raise ValueError("birth_date must be normalizable")
        body = f"440101{normalized_birth.replace('-', '')}{sequence:03d}"[:17]
        return body + "X"


def build_validations(fields: dict[str, Any]) -> list[ValidationResult]:
    validations: list[ValidationResult] = []
    validations.append(_required_validation("required.policyholder_name", fields, "policyholder_name"))
    validations.append(_required_validation("required.policyholder_id_number", fields, "policyholder_id_number"))
    validations.append(_required_validation("required.insured_name", fields, "insured_name"))
    validations.append(_format_validation("format.policyholder_phone", normalize_phone(fields.get("policyholder_phone")), "policyholder_phone"))
    validations.append(_format_validation("format.policyholder_id_number", normalize_id_number(fields.get("policyholder_id_number")), "policyholder_id_number"))
    validations.append(_format_validation("format.application_date", normalize_date(fields.get("application_date")), "application_date"))

    id_number = normalize_id_number(fields.get("insured_id_number") or fields.get("policyholder_id_number"))
    birth_date = normalize_date(fields.get("insured_birth_date"))
    if id_number is None or birth_date is None:
        validations.append(ValidationResult("consistency.birth_date_vs_id_number", "not_applicable", "birth date or id number is missing"))
    else:
        passed = extract_birth_date_from_id_number(id_number) == birth_date
        validations.append(ValidationResult("consistency.birth_date_vs_id_number", "pass" if passed else "fail", "birth date matches id number" if passed else "birth date does not match id number"))

    beneficiary_ratio = fields.get("beneficiary_ratio")
    if beneficiary_ratio is None:
        validations.append(ValidationResult("consistency.beneficiary_ratio_sum", "not_applicable", "beneficiary ratio is missing"))
    else:
        passed = str(beneficiary_ratio).strip().rstrip("%") == "100"
        validations.append(ValidationResult("consistency.beneficiary_ratio_sum", "pass" if passed else "fail", "beneficiary ratios sum to 100" if passed else "beneficiary ratios do not sum to 100"))

    relation = str(fields.get("relation_policyholder_to_insured", "")).strip().lower()
    passed_relation = relation in _RELATIONS
    validations.append(ValidationResult("consistency.policyholder_insured_relation", "pass" if passed_relation else "fail", "relation is allowed" if passed_relation else "relation is not allowed"))

    payment_mode = normalize_known_field("payment_mode", fields.get("payment_mode"))
    payment_period_years = fields.get("payment_period_years")
    combo_ok = payment_mode in _PAYMENT_MODES and not (payment_mode == "single_premium" and payment_period_years not in {1, "1"})
    validations.append(ValidationResult("consistency.product_payment_combo", "pass" if combo_ok else "fail", "product and payment settings are consistent" if combo_ok else "product and payment settings are inconsistent"))

    checkboxes = fields.get("checkboxes", {}) if isinstance(fields.get("checkboxes"), dict) else {}
    selected_modes = sum(1 for key in ("payment_mode.annual", "payment_mode.monthly", "payment_mode.single_premium") if normalize_checkbox_value(checkboxes.get(key)) is True)
    exclusive_ok = selected_modes <= 1
    validations.append(ValidationResult("checkbox.payment_mode_exclusive", "pass" if exclusive_ok else "fail", "payment mode checkboxes are mutually exclusive" if exclusive_ok else "payment mode checkboxes conflict"))

    auto_debit = normalize_checkbox_value(checkboxes.get("payment_mode.auto_debit"))
    if auto_debit is not True:
        validations.append(ValidationResult("checkbox.auto_debit_requires_account", "not_applicable", "auto debit is not selected"))
    else:
        account = fields.get("auto_debit_account") or fields.get("bank_account_number")
        has_account = isinstance(account, str) and bool(account.strip())
        validations.append(ValidationResult("checkbox.auto_debit_requires_account", "pass" if has_account else "fail", "auto debit account is present" if has_account else "auto debit account is missing"))

    return validations


def _required_validation(rule_id: str, fields: dict[str, Any], field_name: str) -> ValidationResult:
    present = field_name in fields and isinstance(fields.get(field_name), str) and bool(str(fields.get(field_name)).strip())
    return ValidationResult(rule_id, "pass" if present else "fail", f"{field_name} is present" if present else f"{field_name} is missing")


def _format_validation(rule_id: str, normalized_value: str | None, field_name: str) -> ValidationResult:
    if normalized_value is None:
        return ValidationResult(rule_id, "not_applicable", f"{field_name} is missing or invalid")
    return ValidationResult(rule_id, "pass", f"{field_name} is normalized")


def apply_ocr_noise(text: str, level: str, rng: random.Random) -> str:
    if level == "low":
        return _replace_once(text, rng, probability=0.15)
    if level == "medium":
        return _replace_once(_replace_once(text, rng, probability=0.3), rng, probability=0.3)
    if level == "high":
        mutated = _replace_once(text, rng, probability=0.55)
        if len(mutated) > 2 and rng.random() < 0.4:
            drop_index = rng.randrange(len(mutated))
            mutated = mutated[:drop_index] + mutated[drop_index + 1 :]
        if rng.random() < 0.4:
            mutated = mutated.replace("", " ").strip()
        return mutated
    return text


def _replace_once(text: str, rng: random.Random, probability: float) -> str:
    replacements = {
        "0": "O",
        "1": "I",
        "8": "B",
        "保": "堡",
        "险": "脸",
        "缴": "激",
        "月": "目",
        "年": "午",
        "张": "弓",
    }
    characters = list(text)
    for index, char in enumerate(characters):
        if char in replacements and rng.random() < probability:
            characters[index] = replacements[char]
    return "".join(characters)


def build_training_record(sample: SyntheticSample, *, task_type: str = "SFT_gold") -> dict[str, Any]:
    if task_type not in TASK_TYPES:
        supported = ", ".join(TASK_TYPES)
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of: {supported}")

    record: dict[str, Any] = {
        "task_type": task_type,
        "input": sample.input.to_dict(),
        "metadata": dict(sample.metadata),
    }
    if task_type != "RL_prompt_only":
        record["reference"] = sample.reference.to_dict()
    return record


def build_sft_record(sample: SyntheticSample, *, task_type: str = "SFT_gold") -> dict[str, Any]:
    return build_training_record(sample, task_type=task_type)


def export_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data for VeriDoc-RL.")
    parser.add_argument("--count", type=int, default=8, help="Number of records to generate.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed.")
    parser.add_argument("--task-type", choices=TASK_TYPES, default="SFT_gold")
    parser.add_argument("--template-family", dest="template_families", choices=TEMPLATE_FAMILIES, action="append")
    parser.add_argument("--ocr-noise-level", dest="ocr_noise_levels", choices=OCR_NOISE_LEVELS, action="append")
    parser.add_argument("--hard-case-type", dest="hard_case_types", choices=HARD_CASE_TYPES, action="append")
    parser.add_argument("--rule-complexity", dest="rule_complexities", choices=RULE_COMPLEXITIES, action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    generator = SyntheticFormGenerator(seed=args.seed)
    samples = generator.generate_dataset(
        count=args.count,
        template_families=args.template_families,
        ocr_noise_levels=args.ocr_noise_levels,
        hard_case_types=args.hard_case_types,
        rule_complexities=args.rule_complexities,
        task_type=args.task_type,
    )
    export_jsonl(args.output_path, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
