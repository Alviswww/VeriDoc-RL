from __future__ import annotations

from veridoc_rl.rewards import (
    RewardComponent,
    aggregate_reward,
    build_reward_components,
    get_reward_weights,
    score_verifier_results,
)
from veridoc_rl.verifiers.base import VerificationResult


def test_aggregate_reward_combines_weighted_scores() -> None:
    payload = aggregate_reward(
        [
            RewardComponent(
                name="schema_reward",
                weight=0.2,
                result=VerificationResult(
                    passed=True,
                    score=1.0,
                    name="schema_reward",
                ),
            ),
            RewardComponent(
                name="field_match_reward",
                weight=0.3,
                result=VerificationResult(
                    passed=False,
                    score=0.5,
                    name="field_match_reward",
                ),
            ),
        ]
    )

    assert payload["total_reward"] == 0.35
    assert payload["raw::field_match_reward"] == 0.5
    assert payload["weighted::schema_reward"] == 0.2


def test_get_reward_weights_supports_ablation_profiles() -> None:
    weights = get_reward_weights(profile="rlvr_without_checkbox_logic")

    assert "checkbox_logic_reward" not in weights
    assert weights["schema_reward"] == 0.2


def test_build_reward_components_filters_disabled_verifiers() -> None:
    components = build_reward_components(
        [
            VerificationResult(passed=True, score=1.0, name="schema_reward"),
            VerificationResult(passed=False, score=0.0, name="checkbox_logic_reward"),
        ],
        profile="rlvr_without_checkbox_logic",
    )

    assert [component.name for component in components] == ["schema_reward"]


def test_score_verifier_results_includes_profile_and_total_reward() -> None:
    payload = score_verifier_results(
        [
            VerificationResult(passed=True, score=1.0, name="schema_reward"),
            VerificationResult(passed=True, score=1.0, name="field_match_reward"),
        ],
        profile="default",
    )

    assert payload["profile"] == "default"
    assert payload["total_reward"] == 0.5
