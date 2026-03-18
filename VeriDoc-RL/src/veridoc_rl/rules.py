from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    category: str
    description: str


RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id="required.policyholder_name",
        category="required",
        description="Policyholder name must be present.",
    ),
    RuleDefinition(
        rule_id="required.policyholder_id_number",
        category="required",
        description="Policyholder ID number must be present.",
    ),
    RuleDefinition(
        rule_id="required.insured_name",
        category="required",
        description="Insured name must be present.",
    ),
    RuleDefinition(
        rule_id="format.policyholder_phone",
        category="format",
        description="Policyholder phone must normalize to an 11-digit number.",
    ),
    RuleDefinition(
        rule_id="format.policyholder_id_number",
        category="format",
        description="Policyholder ID number must satisfy the expected format.",
    ),
    RuleDefinition(
        rule_id="format.application_date",
        category="format",
        description="Application date must normalize to YYYY-MM-DD.",
    ),
    RuleDefinition(
        rule_id="consistency.birth_date_vs_id_number",
        category="consistency",
        description="Birth date must match the birth date encoded in the ID number.",
    ),
    RuleDefinition(
        rule_id="consistency.beneficiary_ratio_sum",
        category="consistency",
        description="Beneficiary ratios must sum to 100.",
    ),
    RuleDefinition(
        rule_id="consistency.policyholder_insured_relation",
        category="consistency",
        description="Policyholder-to-insured relation must be in the allowed set.",
    ),
    RuleDefinition(
        rule_id="consistency.product_payment_combo",
        category="consistency",
        description="Product and payment attributes must form a valid combination.",
    ),
    RuleDefinition(
        rule_id="checkbox.payment_mode_exclusive",
        category="checkbox",
        description="Mutually exclusive payment mode checkboxes must not conflict.",
    ),
    RuleDefinition(
        rule_id="checkbox.auto_debit_requires_account",
        category="checkbox",
        description="Auto-debit selection requires account information.",
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
