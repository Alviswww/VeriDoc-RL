from __future__ import annotations

from veridoc_rl.training.trl_dpo import build_trl_dpo_rows


def test_build_trl_dpo_rows_merges_system_prompt_and_preserves_metadata() -> None:
    rows = build_trl_dpo_rows(
        [
            {
                "sample_id": "sample-1",
                "system_prompt": "Return JSON only.",
                "prompt": "Extract the fields.",
                "chosen": '{"fields":{"policyholder_name":"张三"}}',
                "rejected": '{"fields":{"policyholder_name":""}}',
                "reward_profile": "default",
                "reward_margin": 0.1,
                "metadata": {"bucket": {"template_family": "template_a"}},
            }
        ]
    )

    assert rows[0]["prompt"] == "Return JSON only.\n\nExtract the fields."
    assert rows[0]["chosen"] == '{"fields":{"policyholder_name":"张三"}}'
    assert rows[0]["reward_margin"] == 0.1
    assert '"template_family": "template_a"' in rows[0]["metadata"]
