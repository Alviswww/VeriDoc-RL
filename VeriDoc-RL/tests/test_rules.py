from __future__ import annotations

import pytest

from veridoc_rl.rules import get_rule, has_rule, list_rule_ids, list_rules


def test_rule_registry_lookup() -> None:
    rule = get_rule("必填.投保人姓名")

    assert rule.category == "必填"
    assert "必须存在" in rule.description


def test_rule_registry_filtering() -> None:
    rule_ids = list_rule_ids(category="勾选")

    assert rule_ids == (
        "勾选.缴费方式互斥",
        "勾选.自动扣款需账户",
    )
    assert len(list_rules()) >= len(rule_ids)


def test_has_rule_reflects_registry() -> None:
    assert has_rule("格式.投保人联系电话")
    assert not has_rule("missing.rule")


def test_get_rule_raises_for_unknown_rule() -> None:
    with pytest.raises(KeyError):
        get_rule("missing.rule")
