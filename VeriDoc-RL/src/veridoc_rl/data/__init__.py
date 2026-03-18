from __future__ import annotations

from typing import Any

__all__ = [
    "PreferenceExample",
    "SyntheticFormGenerator",
    "build_preference_pairs",
    "build_sft_record",
    "build_training_record",
    "export_jsonl",
    "export_preference_jsonl",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        exports = {
            "PreferenceExample": _load_preferences_module().PreferenceExample,
            "SyntheticFormGenerator": _load_synthetic_module().SyntheticFormGenerator,
            "build_preference_pairs": _load_preferences_module().build_preference_pairs,
            "build_sft_record": _load_synthetic_module().build_sft_record,
            "build_training_record": _load_synthetic_module().build_training_record,
            "export_jsonl": _load_synthetic_module().export_jsonl,
            "export_preference_jsonl": _load_preferences_module().export_preference_jsonl,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_synthetic_module() -> Any:
    from veridoc_rl.data import synthetic

    return synthetic


def _load_preferences_module() -> Any:
    from veridoc_rl.data import preferences

    return preferences
