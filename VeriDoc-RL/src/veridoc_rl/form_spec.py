from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FIELD_POLICYHOLDER_NAME = "投保人姓名"
FIELD_POLICYHOLDER_GENDER = "投保人性别"
FIELD_POLICYHOLDER_ID_NUMBER = "投保人证件号码"
FIELD_POLICYHOLDER_PHONE = "投保人联系电话"
FIELD_POLICYHOLDER_ADDRESS = "投保人地址"
FIELD_INSURED_NAME = "被保人姓名"
FIELD_INSURED_GENDER = "被保人性别"
FIELD_INSURED_ID_NUMBER = "被保人证件号码"
FIELD_INSURED_BIRTH_DATE = "被保人出生日期"
FIELD_RELATION = "投被保人关系"
FIELD_PRODUCT_NAME = "产品名称"
FIELD_COVERAGE_AMOUNT = "保额"
FIELD_CURRENCY = "币种"
FIELD_PAYMENT_MODE = "缴费方式"
FIELD_PAYMENT_PERIOD_YEARS = "缴费年期"
FIELD_BENEFICIARY_NAME = "受益人姓名"
FIELD_BENEFICIARY_RATIO = "受益比例"
FIELD_SIGNATURE_PRESENT = "是否已签名"
FIELD_APPLICATION_DATE = "申请日期"
FIELD_AUTO_DEBIT_ACCOUNT = "自动扣款账户"
FIELD_CHECKBOXES = "勾选项"

CHECKBOX_PAYMENT_ANNUAL = "缴费方式.年缴"
CHECKBOX_PAYMENT_SEMI_ANNUAL = "缴费方式.半年缴"
CHECKBOX_PAYMENT_QUARTERLY = "缴费方式.季缴"
CHECKBOX_PAYMENT_MONTHLY = "缴费方式.月缴"
CHECKBOX_PAYMENT_SINGLE = "缴费方式.趸缴"
CHECKBOX_AUTO_DEBIT = "缴费方式.自动扣款"

PAYMENT_MODE_ANNUAL = "年缴"
PAYMENT_MODE_SEMI_ANNUAL = "半年缴"
PAYMENT_MODE_QUARTERLY = "季缴"
PAYMENT_MODE_MONTHLY = "月缴"
PAYMENT_MODE_SINGLE = "趸缴"

RELATION_SELF = "本人"
RELATION_SPOUSE = "配偶"
RELATION_CHILD = "子女"
RELATION_PARENT = "父母"
RELATION_EMPLOYEE = "雇员"
RELATION_OTHER = "其他"

RULE_REQUIRED_POLICYHOLDER_NAME = "必填.投保人姓名"
RULE_REQUIRED_POLICYHOLDER_ID_NUMBER = "必填.投保人证件号码"
RULE_REQUIRED_INSURED_NAME = "必填.被保人姓名"
RULE_FORMAT_POLICYHOLDER_PHONE = "格式.投保人联系电话"
RULE_FORMAT_POLICYHOLDER_ID_NUMBER = "格式.投保人证件号码"
RULE_FORMAT_APPLICATION_DATE = "格式.申请日期"
RULE_CONSISTENCY_BIRTH_DATE_VS_ID = "一致性.出生日期证件号码"
RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM = "一致性.受益比例合计"
RULE_CONSISTENCY_RELATION = "一致性.投被保人关系"
RULE_CONSISTENCY_PRODUCT_PAYMENT = "一致性.产品缴费组合"
RULE_CHECKBOX_PAYMENT_MODE_EXCLUSIVE = "勾选.缴费方式互斥"
RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT = "勾选.自动扣款需账户"

ALL_FIELDS: tuple[str, ...] = (
    FIELD_POLICYHOLDER_NAME,
    FIELD_POLICYHOLDER_GENDER,
    FIELD_POLICYHOLDER_ID_NUMBER,
    FIELD_POLICYHOLDER_PHONE,
    FIELD_POLICYHOLDER_ADDRESS,
    FIELD_INSURED_NAME,
    FIELD_INSURED_GENDER,
    FIELD_INSURED_ID_NUMBER,
    FIELD_INSURED_BIRTH_DATE,
    FIELD_RELATION,
    FIELD_PRODUCT_NAME,
    FIELD_COVERAGE_AMOUNT,
    FIELD_CURRENCY,
    FIELD_PAYMENT_MODE,
    FIELD_PAYMENT_PERIOD_YEARS,
    FIELD_BENEFICIARY_NAME,
    FIELD_BENEFICIARY_RATIO,
    FIELD_SIGNATURE_PRESENT,
    FIELD_APPLICATION_DATE,
    FIELD_AUTO_DEBIT_ACCOUNT,
    FIELD_CHECKBOXES,
)

FIELD_ALIASES: dict[str, str] = {
    "policyholder_name": FIELD_POLICYHOLDER_NAME,
    "policyholder_gender": FIELD_POLICYHOLDER_GENDER,
    "policyholder_id_number": FIELD_POLICYHOLDER_ID_NUMBER,
    "policyholder_phone": FIELD_POLICYHOLDER_PHONE,
    "policyholder_address": FIELD_POLICYHOLDER_ADDRESS,
    "insured_name": FIELD_INSURED_NAME,
    "insured_gender": FIELD_INSURED_GENDER,
    "insured_id_number": FIELD_INSURED_ID_NUMBER,
    "insured_birth_date": FIELD_INSURED_BIRTH_DATE,
    "policyholder_birth_date": FIELD_INSURED_BIRTH_DATE,
    "relation_policyholder_to_insured": FIELD_RELATION,
    "product_name": FIELD_PRODUCT_NAME,
    "coverage_amount": FIELD_COVERAGE_AMOUNT,
    "currency": FIELD_CURRENCY,
    "payment_mode": FIELD_PAYMENT_MODE,
    "payment_period_years": FIELD_PAYMENT_PERIOD_YEARS,
    "beneficiary_name": FIELD_BENEFICIARY_NAME,
    "beneficiary_ratio": FIELD_BENEFICIARY_RATIO,
    "signature_present": FIELD_SIGNATURE_PRESENT,
    "application_date": FIELD_APPLICATION_DATE,
    "auto_debit_account": FIELD_AUTO_DEBIT_ACCOUNT,
    "bank_account_number": FIELD_AUTO_DEBIT_ACCOUNT,
    "checkboxes": FIELD_CHECKBOXES,
}

CHECKBOX_ALIASES: dict[str, str] = {
    "payment_mode.annual": CHECKBOX_PAYMENT_ANNUAL,
    "payment_mode.semi_annual": CHECKBOX_PAYMENT_SEMI_ANNUAL,
    "payment_mode.quarterly": CHECKBOX_PAYMENT_QUARTERLY,
    "payment_mode.monthly": CHECKBOX_PAYMENT_MONTHLY,
    "payment_mode.single_premium": CHECKBOX_PAYMENT_SINGLE,
    "payment_mode.auto_debit": CHECKBOX_AUTO_DEBIT,
}

RULE_ID_ALIASES: dict[str, str] = {
    "required.policyholder_name": RULE_REQUIRED_POLICYHOLDER_NAME,
    "required.policyholder_id_number": RULE_REQUIRED_POLICYHOLDER_ID_NUMBER,
    "required.insured_name": RULE_REQUIRED_INSURED_NAME,
    "format.policyholder_phone": RULE_FORMAT_POLICYHOLDER_PHONE,
    "format.policyholder_id_number": RULE_FORMAT_POLICYHOLDER_ID_NUMBER,
    "format.application_date": RULE_FORMAT_APPLICATION_DATE,
    "consistency.birth_date_vs_id_number": RULE_CONSISTENCY_BIRTH_DATE_VS_ID,
    "consistency.beneficiary_ratio_sum": RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM,
    "consistency.policyholder_insured_relation": RULE_CONSISTENCY_RELATION,
    "consistency.product_payment_combo": RULE_CONSISTENCY_PRODUCT_PAYMENT,
    "checkbox.payment_mode_exclusive": RULE_CHECKBOX_PAYMENT_MODE_EXCLUSIVE,
    "checkbox.auto_debit_requires_account": RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT,
}

PAYMENT_MODE_ALIASES: dict[str, str] = {
    "annual": PAYMENT_MODE_ANNUAL,
    "年缴": PAYMENT_MODE_ANNUAL,
    "yearly": PAYMENT_MODE_ANNUAL,
    "semi_annual": PAYMENT_MODE_SEMI_ANNUAL,
    "半年缴": PAYMENT_MODE_SEMI_ANNUAL,
    "quarterly": PAYMENT_MODE_QUARTERLY,
    "季缴": PAYMENT_MODE_QUARTERLY,
    "monthly": PAYMENT_MODE_MONTHLY,
    "月缴": PAYMENT_MODE_MONTHLY,
    "single_premium": PAYMENT_MODE_SINGLE,
    "趸缴": PAYMENT_MODE_SINGLE,
}

RELATION_ALIASES: dict[str, str] = {
    "self": RELATION_SELF,
    "本人": RELATION_SELF,
    "spouse": RELATION_SPOUSE,
    "配偶": RELATION_SPOUSE,
    "child": RELATION_CHILD,
    "子女": RELATION_CHILD,
    "parent": RELATION_PARENT,
    "父母": RELATION_PARENT,
    "employee": RELATION_EMPLOYEE,
    "雇员": RELATION_EMPLOYEE,
    "other": RELATION_OTHER,
    "其他": RELATION_OTHER,
}

GENDER_ALIASES: dict[str, str] = {
    "male": "男",
    "男": "男",
    "female": "女",
    "女": "女",
}

PRODUCT_NAME_ALIASES: dict[str, str] = {
    "whole_life": "终身寿险",
    "终身寿险": "终身寿险",
    "critical_illness": "重疾险",
    "重疾险": "重疾险",
    "medical": "医疗险",
    "医疗险": "医疗险",
}


def canonicalize_field_name(field_name: str) -> str:
    return FIELD_ALIASES.get(field_name, field_name)


def canonicalize_checkbox_name(field_name: str) -> str:
    return CHECKBOX_ALIASES.get(field_name, field_name)


def canonicalize_rule_id(rule_id: str) -> str:
    return RULE_ID_ALIASES.get(rule_id, rule_id)


def canonicalize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for raw_key, value in fields.items():
        key = canonicalize_field_name(str(raw_key))
        if key == FIELD_CHECKBOXES and isinstance(value, Mapping):
            canonical[key] = {
                canonicalize_checkbox_name(str(item_key)): item_value
                for item_key, item_value in value.items()
            }
            continue
        canonical[key] = value
    return canonical


def canonicalize_validations(validations: Any) -> list[dict[str, Any]]:
    if not isinstance(validations, list):
        return []
    canonical: list[dict[str, Any]] = []
    for item in validations:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        if "rule_id" in row:
            row["rule_id"] = canonicalize_rule_id(str(row["rule_id"]))
        canonical.append(row)
    return canonical


def canonicalize_prediction_payload(payload: Mapping[str, Any], *, sample_id: str) -> dict[str, Any]:
    fields = payload.get("fields", {})
    return {
        "sample_id": payload.get("sample_id", sample_id),
        "fields": canonicalize_fields(fields) if isinstance(fields, Mapping) else {},
        "validations": canonicalize_validations(payload.get("validations")),
    }
