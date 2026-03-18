from __future__ import annotations

import json

from veridoc_rl.smoke import load_fixture, main, run_smoke


def test_run_smoke_returns_verification_payload() -> None:
    fixture = load_fixture()

    payload = run_smoke(fixture=fixture, prediction=fixture["prediction"])

    assert payload["reward"]["total_reward"] > 0
    assert payload["verification"]["schema_reward"]["passed"] is True


def test_cli_main_prints_json(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert "metrics" in payload
