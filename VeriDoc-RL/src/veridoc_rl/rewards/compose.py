from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from veridoc_rl.verifiers.base import VerificationResult


DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "schema_reward": 0.20,
    "field_match_reward": 0.30,
    "normalization_reward": 0.15,
    "cross_field_consistency_reward": 0.20,
    "checkbox_logic_reward": 0.10,
    "ocr_robustness_reward": 0.05,
}
REWARD_PROFILES: dict[str, dict[str, float]] = {
    "default": dict(DEFAULT_REWARD_WEIGHTS),
    "rlvr": dict(DEFAULT_REWARD_WEIGHTS),
    "rlvr_without_cross_field_consistency": {
        name: weight
        for name, weight in DEFAULT_REWARD_WEIGHTS.items()
        if name != "cross_field_consistency_reward"
    },
    "rlvr_without_checkbox_logic": {
        name: weight
        for name, weight in DEFAULT_REWARD_WEIGHTS.items()
        if name != "checkbox_logic_reward"
    },
}


@dataclass
class RewardComponent:
    name: str
    weight: float
    result: VerificationResult

    @property
    def weighted_score(self) -> float:
        return self.weight * self.result.score


def aggregate_reward(components: Iterable[RewardComponent]) -> dict[str, float]:
    items = list(components)
    total = sum(item.weighted_score for item in items)
    raw = {item.name: item.result.score for item in items}
    weighted = {item.name: item.weighted_score for item in items}
    return {
        "total_reward": total,
        "component_count": float(len(items)),
        **{f"raw::{k}": v for k, v in raw.items()},
        **{f"weighted::{k}": v for k, v in weighted.items()},
    }


def get_reward_weights(
    profile: str = "default",
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    try:
        weights = dict(REWARD_PROFILES[profile])
    except KeyError as exc:
        available = ", ".join(sorted(REWARD_PROFILES))
        raise KeyError(f"Unknown reward profile: {profile}. Available profiles: {available}") from exc
    if overrides:
        weights.update({name: float(weight) for name, weight in overrides.items()})
    return weights


def list_reward_profiles() -> tuple[str, ...]:
    return tuple(sorted(REWARD_PROFILES))


def build_reward_components(
    results: Iterable[VerificationResult],
    *,
    profile: str = "default",
    overrides: Mapping[str, float] | None = None,
) -> list[RewardComponent]:
    weights = get_reward_weights(profile=profile, overrides=overrides)
    return [
        RewardComponent(name=result.name, weight=weights[result.name], result=result)
        for result in results
        if result.name in weights and weights[result.name] > 0
    ]


def score_verifier_results(
    results: Iterable[VerificationResult],
    *,
    profile: str = "default",
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float | str]:
    payload = aggregate_reward(
        build_reward_components(results, profile=profile, overrides=overrides)
    )
    payload["profile"] = profile
    return payload
