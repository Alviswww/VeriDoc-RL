from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from veridoc_rl.form_spec import (
    CHECKBOX_AUTO_DEBIT,
    CHECKBOX_PAYMENT_ANNUAL,
    CHECKBOX_PAYMENT_MONTHLY,
    CHECKBOX_PAYMENT_SINGLE,
    FIELD_APPLICATION_DATE,
    FIELD_AUTO_DEBIT_ACCOUNT,
    FIELD_BENEFICIARY_NAME,
    FIELD_BENEFICIARY_RATIO,
    FIELD_CHECKBOXES,
    FIELD_COVERAGE_AMOUNT,
    FIELD_CURRENCY,
    FIELD_INSURED_BIRTH_DATE,
    FIELD_INSURED_GENDER,
    FIELD_INSURED_ID_NUMBER,
    FIELD_INSURED_NAME,
    FIELD_PAYMENT_MODE,
    FIELD_PAYMENT_PERIOD_YEARS,
    FIELD_POLICYHOLDER_ADDRESS,
    FIELD_POLICYHOLDER_GENDER,
    FIELD_POLICYHOLDER_ID_NUMBER,
    FIELD_POLICYHOLDER_NAME,
    FIELD_POLICYHOLDER_PHONE,
    FIELD_PRODUCT_NAME,
    FIELD_RELATION,
    FIELD_SIGNATURE_PRESENT,
    PAYMENT_MODE_ANNUAL,
    PAYMENT_MODE_MONTHLY,
    PAYMENT_MODE_SINGLE,
    RELATION_CHILD,
    RELATION_PARENT,
    RELATION_SELF,
    RELATION_SPOUSE,
    RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT,
    RULE_CHECKBOX_PAYMENT_MODE_EXCLUSIVE,
    RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM,
    RULE_CONSISTENCY_BIRTH_DATE_VS_ID,
    RULE_CONSISTENCY_PRODUCT_PAYMENT,
    RULE_CONSISTENCY_RELATION,
    RULE_FORMAT_APPLICATION_DATE,
    RULE_FORMAT_POLICYHOLDER_ID_NUMBER,
    RULE_FORMAT_POLICYHOLDER_PHONE,
    RULE_REQUIRED_INSURED_NAME,
    RULE_REQUIRED_POLICYHOLDER_ID_NUMBER,
    RULE_REQUIRED_POLICYHOLDER_NAME,
)
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
_PRODUCTS = ("终身寿险", "重疾险", "医疗险")
_RELATIONS = (RELATION_SELF, RELATION_SPOUSE, RELATION_CHILD, RELATION_PARENT)
_PAYMENT_MODES = (PAYMENT_MODE_ANNUAL, PAYMENT_MODE_MONTHLY, PAYMENT_MODE_SINGLE)
_CURRENCY = ("CNY", "USD")

_FIELD_LAYOUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "template_a": (
        (FIELD_POLICYHOLDER_NAME, "投保人姓名"),
        (FIELD_POLICYHOLDER_PHONE, "联系电话"),
        (FIELD_POLICYHOLDER_ADDRESS, "联系地址"),
        (FIELD_POLICYHOLDER_ID_NUMBER, "证件号码"),
        (FIELD_INSURED_NAME, "被保人姓名"),
        (FIELD_INSURED_BIRTH_DATE, "出生日期"),
        (FIELD_PRODUCT_NAME, "险种名称"),
        (FIELD_COVERAGE_AMOUNT, "保额"),
        (FIELD_PAYMENT_MODE, "缴费方式"),
        (FIELD_AUTO_DEBIT_ACCOUNT, "自动扣款账户"),
        (FIELD_APPLICATION_DATE, "申请日期"),
    ),
    "template_b": (
        (FIELD_POLICYHOLDER_NAME, "投保人"),
        (FIELD_POLICYHOLDER_PHONE, "手机号"),
        (FIELD_POLICYHOLDER_ADDRESS, "地址"),
        (FIELD_POLICYHOLDER_ID_NUMBER, "身份证号"),
        (FIELD_INSURED_NAME, "被保险人"),
        (FIELD_INSURED_BIRTH_DATE, "生日"),
        (FIELD_PRODUCT_NAME, "产品"),
        (FIELD_COVERAGE_AMOUNT, "保险金额"),
        (FIELD_PAYMENT_MODE, "支付方式"),
        (FIELD_AUTO_DEBIT_ACCOUNT, "扣款账号"),
        (FIELD_APPLICATION_DATE, "投保日期"),
    ),
    "template_c": (
        (FIELD_POLICYHOLDER_NAME, "姓名(投保人)"),
        (FIELD_POLICYHOLDER_PHONE, "手机"),
        (FIELD_POLICYHOLDER_ADDRESS, "联系住址"),
        (FIELD_POLICYHOLDER_ID_NUMBER, "证件号"),
        (FIELD_INSURED_NAME, "姓名(被保人)"),
        (FIELD_INSURED_BIRTH_DATE, "出生年月日"),
        (FIELD_PRODUCT_NAME, "产品计划"),
        (FIELD_COVERAGE_AMOUNT, "保障额度"),
        (FIELD_PAYMENT_MODE, "缴费周期"),
        (FIELD_AUTO_DEBIT_ACCOUNT, "代扣账户"),
        (FIELD_APPLICATION_DATE, "填写日期"),
    ),
}

_OCCLUDED_FIELDS = {FIELD_COVERAGE_AMOUNT, FIELD_APPLICATION_DATE, FIELD_POLICYHOLDER_ADDRESS}
_SMUDGED_FIELDS = {FIELD_POLICYHOLDER_NAME, FIELD_PRODUCT_NAME}


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

        source_fields = self._build_source_fields(sample_index=sample_index, hard_case_type=hard_case_type)
        visible_fields, ocr_tokens = self._build_visible_fields_and_tokens(
            source_fields=source_fields,
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
        validations = build_validations(visible_fields)
        form_output = FormOutput(sample_id=sample_id, fields=visible_fields, validations=validations)
        _assert_reference_supported_by_visible_fields(
            reference_fields=form_output.fields,
            visible_fields=visible_fields,
            sample_id=sample_id,
        )
        metadata = {
            "bucket": {
                "template_family": template_family,
                "ocr_noise_level": ocr_noise_level,
                "hard_case_type": hard_case_type,
                "rule_complexity": rule_complexity,
            },
            "generator_seed": self._seed,
            "sample_index": sample_index,
            "debug": {
                "visible_fields": sorted(key for key in visible_fields if key != FIELD_CHECKBOXES),
                "omitted_fields": sorted(
                    key for key in source_fields if key not in visible_fields and key != FIELD_CHECKBOXES
                ),
            },
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

    def _build_source_fields(self, sample_index: int, hard_case_type: str) -> dict[str, Any]:
        insured_name = self._rng.choice(_NAME_POOL)
        policyholder_name = insured_name if sample_index % 2 == 0 else self._rng.choice(_NAME_POOL)
        birth_date = f"19{90 + sample_index % 10:02d}-{(sample_index % 12) + 1:02d}-{(sample_index % 27) + 1:02d}"
        id_number = self._build_id_number(birth_date=birth_date, sequence=120 + sample_index)
        payment_mode = _PAYMENT_MODES[sample_index % len(_PAYMENT_MODES)]
        relation = RELATION_SELF if policyholder_name == insured_name else _RELATIONS[(sample_index % (len(_RELATIONS) - 1)) + 1]

        fields: dict[str, Any] = {
            FIELD_POLICYHOLDER_NAME: policyholder_name,
            FIELD_POLICYHOLDER_GENDER: "男" if sample_index % 2 == 0 else "女",
            FIELD_POLICYHOLDER_ID_NUMBER: id_number,
            FIELD_POLICYHOLDER_PHONE: f"1380013{sample_index % 10000:04d}",
            FIELD_POLICYHOLDER_ADDRESS: _ADDRESS_POOL[sample_index % len(_ADDRESS_POOL)],
            FIELD_INSURED_NAME: insured_name,
            FIELD_INSURED_GENDER: "男" if sample_index % 2 == 0 else "女",
            FIELD_INSURED_ID_NUMBER: id_number,
            FIELD_INSURED_BIRTH_DATE: birth_date,
            FIELD_RELATION: relation,
            FIELD_PRODUCT_NAME: _PRODUCTS[sample_index % len(_PRODUCTS)],
            FIELD_COVERAGE_AMOUNT: str(100000 * ((sample_index % 5) + 1)),
            FIELD_CURRENCY: _CURRENCY[sample_index % len(_CURRENCY)],
            FIELD_PAYMENT_MODE: payment_mode,
            FIELD_PAYMENT_PERIOD_YEARS: 1 if payment_mode == PAYMENT_MODE_SINGLE else 20,
            FIELD_BENEFICIARY_NAME: self._rng.choice(_NAME_POOL),
            FIELD_BENEFICIARY_RATIO: "100",
            FIELD_SIGNATURE_PRESENT: True,
            FIELD_APPLICATION_DATE: f"2024-{((sample_index + 1) % 12) + 1:02d}-{((sample_index + 9) % 27) + 1:02d}",
            FIELD_CHECKBOXES: {
                CHECKBOX_PAYMENT_ANNUAL: payment_mode == PAYMENT_MODE_ANNUAL,
                CHECKBOX_PAYMENT_MONTHLY: payment_mode == PAYMENT_MODE_MONTHLY,
                CHECKBOX_PAYMENT_SINGLE: payment_mode == PAYMENT_MODE_SINGLE,
                CHECKBOX_AUTO_DEBIT: sample_index % 3 == 0,
            },
        }

        if fields[FIELD_CHECKBOXES][CHECKBOX_AUTO_DEBIT]:
            fields[FIELD_AUTO_DEBIT_ACCOUNT] = f"622202{sample_index % 10000000000:010d}"

        if hard_case_type == "field_missing":
            fields.pop(FIELD_POLICYHOLDER_PHONE, None)
        elif hard_case_type == "checkbox_conflict":
            fields[FIELD_CHECKBOXES][CHECKBOX_PAYMENT_ANNUAL] = True
            fields[FIELD_CHECKBOXES][CHECKBOX_PAYMENT_MONTHLY] = True
        return fields

    def _build_visible_fields_and_tokens(
        self,
        *,
        source_fields: dict[str, Any],
        template_family: str,
        ocr_noise_level: str,
        hard_case_type: str,
    ) -> tuple[dict[str, Any], list[OCRToken]]:
        visible_fields: dict[str, Any] = {}
        tokens: list[OCRToken] = []
        base_x = _TEMPLATE_BASE_X[template_family]
        base_y = _TEMPLATE_BASE_Y[template_family]

        for index, (field_name, label) in enumerate(_FIELD_LAYOUTS[template_family]):
            y = base_y + index * 34
            tokens.append(OCRToken(text=label, bbox=[base_x, y, base_x + 90, y + 18], page=1))
            if field_name not in source_fields:
                continue
            if hard_case_type == "occluded_region" and field_name in _OCCLUDED_FIELDS:
                continue
            visible_fields[field_name] = source_fields[field_name]
            value_text = self._serialize_value(field_name, source_fields[field_name])
            if hard_case_type == "smudged_text" and field_name in _SMUDGED_FIELDS:
                value_text = apply_ocr_noise(value_text, "high", self._rng)
            else:
                value_text = apply_ocr_noise(value_text, ocr_noise_level, self._rng)
            tokens.append(
                OCRToken(
                    text=value_text,
                    bbox=[base_x + 110, y, base_x + 320, y + 18],
                    page=1,
                )
            )

        checkboxes = source_fields.get(FIELD_CHECKBOXES, {})
        if isinstance(checkboxes, dict):
            visible_fields[FIELD_CHECKBOXES] = dict(checkboxes)
        checkbox_text = self._serialize_checkboxes(checkboxes)
        tokens.append(
            OCRToken(
                text=apply_ocr_noise(checkbox_text, ocr_noise_level, self._rng),
                bbox=[base_x, base_y + 420, base_x + 320, base_y + 440],
                page=1,
            )
        )
        return visible_fields, tokens

    def _serialize_value(self, field_name: str, value: Any) -> str:
        if field_name == FIELD_COVERAGE_AMOUNT:
            return f"{value}元"
        if field_name == FIELD_APPLICATION_DATE:
            normalized = normalize_date(value) or str(value)
            return normalized.replace("-", "/")
        return str(value)

    def _serialize_checkboxes(self, checkboxes: Any) -> str:
        if not isinstance(checkboxes, dict):
            return ""
        annual = "√" if checkboxes.get(CHECKBOX_PAYMENT_ANNUAL) else "□"
        monthly = "√" if checkboxes.get(CHECKBOX_PAYMENT_MONTHLY) else "□"
        single = "√" if checkboxes.get(CHECKBOX_PAYMENT_SINGLE) else "□"
        auto_debit = "√" if checkboxes.get(CHECKBOX_AUTO_DEBIT) else "□"
        return f"年缴{annual} 月缴{monthly} 趸缴{single} 自动扣款{auto_debit}"

    def _build_id_number(self, birth_date: str, sequence: int) -> str:
        normalized_birth = normalize_date(birth_date)
        if normalized_birth is None:
            raise ValueError("birth_date must be normalizable")
        body = f"440101{normalized_birth.replace('-', '')}{sequence:03d}"[:17]
        return body + "X"


def build_validations(fields: dict[str, Any]) -> list[ValidationResult]:
    validations: list[ValidationResult] = []
    validations.append(_required_validation(RULE_REQUIRED_POLICYHOLDER_NAME, fields, FIELD_POLICYHOLDER_NAME))
    validations.append(_required_validation(RULE_REQUIRED_POLICYHOLDER_ID_NUMBER, fields, FIELD_POLICYHOLDER_ID_NUMBER))
    validations.append(_required_validation(RULE_REQUIRED_INSURED_NAME, fields, FIELD_INSURED_NAME))
    validations.append(_format_validation(RULE_FORMAT_POLICYHOLDER_PHONE, normalize_phone(fields.get(FIELD_POLICYHOLDER_PHONE)), FIELD_POLICYHOLDER_PHONE))
    validations.append(_format_validation(RULE_FORMAT_POLICYHOLDER_ID_NUMBER, normalize_id_number(fields.get(FIELD_POLICYHOLDER_ID_NUMBER)), FIELD_POLICYHOLDER_ID_NUMBER))
    validations.append(_format_validation(RULE_FORMAT_APPLICATION_DATE, normalize_date(fields.get(FIELD_APPLICATION_DATE)), FIELD_APPLICATION_DATE))

    id_number = normalize_id_number(fields.get(FIELD_INSURED_ID_NUMBER) or fields.get(FIELD_POLICYHOLDER_ID_NUMBER))
    birth_date = normalize_date(fields.get(FIELD_INSURED_BIRTH_DATE))
    if id_number is None or birth_date is None:
        validations.append(ValidationResult(RULE_CONSISTENCY_BIRTH_DATE_VS_ID, "not_applicable", "出生日期或证件号码缺失，无法校验。"))
    else:
        passed = extract_birth_date_from_id_number(id_number) == birth_date
        validations.append(ValidationResult(RULE_CONSISTENCY_BIRTH_DATE_VS_ID, "pass" if passed else "fail", "出生日期与证件号码一致。" if passed else "出生日期与证件号码不一致。"))

    beneficiary_ratio = fields.get(FIELD_BENEFICIARY_RATIO)
    if beneficiary_ratio is None:
        validations.append(ValidationResult(RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM, "not_applicable", "受益比例缺失，无法校验。"))
    else:
        passed = str(beneficiary_ratio).strip().rstrip("%") == "100"
        validations.append(ValidationResult(RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM, "pass" if passed else "fail", "受益比例合计为 100。" if passed else "受益比例合计不为 100。"))

    relation = normalize_known_field(FIELD_RELATION, fields.get(FIELD_RELATION))
    passed_relation = relation in _RELATIONS
    validations.append(ValidationResult(RULE_CONSISTENCY_RELATION, "pass" if passed_relation else "fail", "投被保人关系合法。" if passed_relation else "投被保人关系不合法。"))

    payment_mode = normalize_known_field(FIELD_PAYMENT_MODE, fields.get(FIELD_PAYMENT_MODE))
    payment_period_years = fields.get(FIELD_PAYMENT_PERIOD_YEARS)
    combo_ok = payment_mode in _PAYMENT_MODES and not (payment_mode == PAYMENT_MODE_SINGLE and payment_period_years not in {1, "1"})
    validations.append(ValidationResult(RULE_CONSISTENCY_PRODUCT_PAYMENT, "pass" if combo_ok else "fail", "产品和缴费设置一致。" if combo_ok else "产品和缴费设置不一致。"))

    checkboxes = fields.get(FIELD_CHECKBOXES, {}) if isinstance(fields.get(FIELD_CHECKBOXES), dict) else {}
    selected_modes = sum(1 for key in (CHECKBOX_PAYMENT_ANNUAL, CHECKBOX_PAYMENT_MONTHLY, CHECKBOX_PAYMENT_SINGLE) if normalize_checkbox_value(checkboxes.get(key)) is True)
    exclusive_ok = selected_modes <= 1
    validations.append(ValidationResult(RULE_CHECKBOX_PAYMENT_MODE_EXCLUSIVE, "pass" if exclusive_ok else "fail", "缴费方式勾选互斥。" if exclusive_ok else "缴费方式勾选冲突。"))

    auto_debit = normalize_checkbox_value(checkboxes.get(CHECKBOX_AUTO_DEBIT))
    if auto_debit is not True:
        validations.append(ValidationResult(RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT, "not_applicable", "未勾选自动扣款。"))
    else:
        account = fields.get(FIELD_AUTO_DEBIT_ACCOUNT)
        has_account = isinstance(account, str) and bool(account.strip())
        validations.append(ValidationResult(RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT, "pass" if has_account else "fail", "自动扣款账户已填写。" if has_account else "勾选自动扣款但未填写账户。"))

    return validations


def _required_validation(rule_id: str, fields: dict[str, Any], field_name: str) -> ValidationResult:
    present = field_name in fields and isinstance(fields.get(field_name), str) and bool(str(fields.get(field_name)).strip())
    return ValidationResult(rule_id, "pass" if present else "fail", f"{field_name}已填写。" if present else f"{field_name}缺失。")


def _format_validation(rule_id: str, normalized_value: str | None, field_name: str) -> ValidationResult:
    if normalized_value is None:
        return ValidationResult(rule_id, "not_applicable", f"{field_name}缺失或格式不合法。")
    return ValidationResult(rule_id, "pass", f"{field_name}格式合法。")


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
        "地": "她",
    }
    characters = list(text)
    for index, char in enumerate(characters):
        if char in replacements and rng.random() < probability:
            characters[index] = replacements[char]
    return "".join(characters)


def _assert_reference_supported_by_visible_fields(
    *,
    reference_fields: dict[str, Any],
    visible_fields: dict[str, Any],
    sample_id: str,
) -> None:
    for field_name, reference_value in reference_fields.items():
        if field_name not in visible_fields:
            raise ValueError(f"{sample_id}: ground truth field `{field_name}` is not visible in OCR source.")
        visible_value = visible_fields[field_name]
        if field_name == FIELD_CHECKBOXES:
            if dict(reference_value) != dict(visible_value):
                raise ValueError(f"{sample_id}: checkbox ground truth does not match visible OCR state.")
            continue
        if reference_value != visible_value:
            raise ValueError(f"{sample_id}: ground truth field `{field_name}` does not match visible OCR source.")


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
