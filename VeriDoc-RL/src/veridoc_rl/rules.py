from __future__ import annotations

from dataclasses import dataclass

from veridoc_rl.form_spec import (
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


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    category: str
    description: str


RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id=RULE_REQUIRED_POLICYHOLDER_NAME,
        category="必填",
        description="投保人姓名必须存在。",
    ),
    RuleDefinition(
        rule_id=RULE_REQUIRED_POLICYHOLDER_ID_NUMBER,
        category="必填",
        description="投保人证件号码必须存在。",
    ),
    RuleDefinition(
        rule_id=RULE_REQUIRED_INSURED_NAME,
        category="必填",
        description="被保人姓名必须存在。",
    ),
    RuleDefinition(
        rule_id=RULE_FORMAT_POLICYHOLDER_PHONE,
        category="格式",
        description="投保人联系电话必须可标准化为 11 位手机号。",
    ),
    RuleDefinition(
        rule_id=RULE_FORMAT_POLICYHOLDER_ID_NUMBER,
        category="格式",
        description="投保人证件号码必须满足合法格式。",
    ),
    RuleDefinition(
        rule_id=RULE_FORMAT_APPLICATION_DATE,
        category="格式",
        description="申请日期必须可标准化为 YYYY-MM-DD。",
    ),
    RuleDefinition(
        rule_id=RULE_CONSISTENCY_BIRTH_DATE_VS_ID,
        category="一致性",
        description="被保人出生日期必须与证件号码中的生日一致。",
    ),
    RuleDefinition(
        rule_id=RULE_CONSISTENCY_BENEFICIARY_RATIO_SUM,
        category="一致性",
        description="受益比例合计必须为 100。",
    ),
    RuleDefinition(
        rule_id=RULE_CONSISTENCY_RELATION,
        category="一致性",
        description="投被保人关系必须落在允许集合中。",
    ),
    RuleDefinition(
        rule_id=RULE_CONSISTENCY_PRODUCT_PAYMENT,
        category="一致性",
        description="产品和缴费属性必须组成有效组合。",
    ),
    RuleDefinition(
        rule_id=RULE_CHECKBOX_PAYMENT_MODE_EXCLUSIVE,
        category="勾选",
        description="缴费方式勾选项必须互斥，不可同时冲突。",
    ),
    RuleDefinition(
        rule_id=RULE_CHECKBOX_AUTO_DEBIT_REQUIRES_ACCOUNT,
        category="勾选",
        description="若勾选自动扣款，必须填写扣款账户。",
    ),
)

RULES_BY_ID: dict[str, RuleDefinition] = {rule.rule_id: rule for rule in RULES}


def get_rule(rule_id: str) -> RuleDefinition:
    return RULES_BY_ID[rule_id]


def list_rules(category: str | None = None) -> tuple[RuleDefinition, ...]:
    if category is None:
        return RULES
    return tuple(rule for rule in RULES if rule.category == category)


def list_rule_ids(category: str | None = None) -> tuple[str, ...]:
    return tuple(rule.rule_id for rule in list_rules(category=category))


def has_rule(rule_id: str) -> bool:
    return rule_id in RULES_BY_ID
