from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class VerificationResult:
    """Structured verifier output for a single model response."""

    passed: bool
    score: float
    name: str
    details: dict[str, Any] = field(default_factory=dict)


class BaseVerifier(ABC):
    """Base interface for verifier components used in RLVR reward shaping."""

    name = "base_verifier"

    @abstractmethod
    def verify(
        self,
        prediction: Mapping[str, Any],
        reference: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        """Return a structured verification result for a single prediction."""
