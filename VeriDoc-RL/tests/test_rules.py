from __future__ import annotations

import pytest

from veridoc_rl.rules import get_rule, has_rule, list_rule_ids, list_rules


def test_rule_registry_lookup() -> None:
    rule = get_rule("required.policyholder_name")

    assert rule.category == "required"
    assert "must be present" in rule.description


def test_rule_registry_filtering() -> None:
    rule_ids = list_rule_ids(category="checkbox")

    assert rule_ids == (
        "checkbox.payment_mode_exclusive",
        "checkbox.auto_debit_requires_account",
    )
    assert len(list_rules()) >= len(rule_ids)


def test_has_rule_reflects_registry() -> None:
    assert has_rule("format.policyholder_phone")
    assert not has_rule("missing.rule")


def test_get_rule_raises_for_unknown_rule() -> None:
    with pytest.raises(KeyError):
        get_rule("missing.rule")
