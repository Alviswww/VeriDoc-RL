from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from veridoc_rl.evaluation.metrics import evaluate_prediction
from veridoc_rl.rewards import score_verifier_results
from veridoc_rl.verifiers import (
    CheckboxLogicVerifier,
    CrossFieldConsistencyVerifier,
    FieldMatchVerifier,
    NormalizationVerifier,
    OCRRobustnessVerifier,
    SchemaVerifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VeriDoc-RL smoke verification.")
    parser.add_argument("--fixture-path", type=Path, help="Path to a JSON fixture file.")
    parser.add_argument("--prediction-path", type=Path, help="Path to a JSON prediction file.")
    parser.add_argument("--prediction-json", help="Inline JSON string for a prediction payload.")
    return parser


def load_fixture(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    default_resource = resources.files("veridoc_rl.fixtures").joinpath("minimal_form_fixture.json")
    return json.loads(default_resource.read_text(encoding="utf-8-sig"))


def load_prediction(args: argparse.Namespace, fixture: Mapping[str, Any]) -> dict[str, Any]:
    if args.prediction_json:
        return json.loads(args.prediction_json)
    if args.prediction_path:
        return json.loads(args.prediction_path.read_text(encoding="utf-8-sig"))

    fixture_prediction = fixture.get("prediction")
    if not isinstance(fixture_prediction, Mapping):
        raise ValueError("Fixture must contain a 'prediction' mapping.")
    return dict(fixture_prediction)


def run_smoke(
    fixture: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    reference = fixture.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("Fixture must contain a 'reference' mapping.")

    context = fixture.get("context", {})
    if not isinstance(context, Mapping):
        raise ValueError("Fixture context must be a mapping when provided.")

    verifiers = [
        SchemaVerifier(),
        FieldMatchVerifier(),
        NormalizationVerifier(),
        CrossFieldConsistencyVerifier(),
        CheckboxLogicVerifier(),
        OCRRobustnessVerifier(),
    ]

    verification_results: dict[str, dict[str, Any]] = {}
    raw_results = []
    for verifier in verifiers:
        result = verifier.verify(prediction=prediction, reference=reference, context=context)
        raw_results.append(result)
        verification_results[result.name] = {
            "passed": result.passed,
            "score": result.score,
            "details": result.details,
        }

    return {
        "verification": verification_results,
        "reward": score_verifier_results(raw_results),
        "metrics": evaluate_prediction(prediction=prediction, reference=reference),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    fixture = load_fixture(path=args.fixture_path)
    prediction = load_prediction(args=args, fixture=fixture)
    payload = run_smoke(fixture=fixture, prediction=prediction)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
