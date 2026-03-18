from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from veridoc_rl.verifiers.base import BaseVerifier, VerificationResult
from veridoc_rl.verifiers.form import (
    CheckboxLogicVerifier,
    CrossFieldConsistencyVerifier,
    FieldMatchVerifier,
    NormalizationVerifier,
    OCRRobustnessVerifier,
    SchemaVerifier,
)


DEFAULT_VERIFIER_CLASSES = (
    SchemaVerifier,
    FieldMatchVerifier,
    NormalizationVerifier,
    CrossFieldConsistencyVerifier,
    CheckboxLogicVerifier,
    OCRRobustnessVerifier,
)


def build_default_verifiers() -> tuple[BaseVerifier, ...]:
    """Instantiate the default verifier suite used by rewards and reporting."""

    return tuple(verifier_class() for verifier_class in DEFAULT_VERIFIER_CLASSES)


def run_verifier_suite(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    verifiers: Sequence[BaseVerifier] | None = None,
) -> list[VerificationResult]:
    """Run a verifier suite and return the structured results in execution order."""

    active_verifiers = tuple(verifiers) if verifiers is not None else build_default_verifiers()
    return [
        verifier.verify(prediction=prediction, reference=reference, context=context)
        for verifier in active_verifiers
    ]


__all__ = [
    "BaseVerifier",
    "VerificationResult",
    "SchemaVerifier",
    "FieldMatchVerifier",
    "NormalizationVerifier",
    "CrossFieldConsistencyVerifier",
    "CheckboxLogicVerifier",
    "OCRRobustnessVerifier",
    "build_default_verifiers",
    "run_verifier_suite",
]
