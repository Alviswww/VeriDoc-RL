from __future__ import annotations

import json
from pathlib import Path

from veridoc_rl.data.synthetic import SyntheticFormGenerator
from veridoc_rl.experiments import load_experiment_matrix
from veridoc_rl.training.corpus import (
    main as prepare_training_main,
)
from veridoc_rl.training.corpus import (
    prepare_dpo_corpus,
    prepare_rl_corpus,
    prepare_sft_corpus,
)
from veridoc_rl.training.manifests import (
    build_training_manifests,
)
from veridoc_rl.training.manifests import (
    main as generate_manifests_main,
)
from veridoc_rl.training.prompting import build_chat_messages, build_user_prompt


def test_build_user_prompt_contains_schema_and_tokens() -> None:
    sample = SyntheticFormGenerator(seed=7).generate_sample(sample_index=0).to_record()

    prompt = build_user_prompt(sample["input"])

    assert "OCR tokens:" in prompt
    assert '"validations"' in prompt
    assert "sample_id:" in prompt


def test_build_chat_messages_includes_assistant_reference() -> None:
    sample = SyntheticFormGenerator(seed=7).generate_sample(sample_index=0).to_record()

    messages = build_chat_messages(sample["input"], reference_payload=sample["reference"])

    assert [item["role"] for item in messages] == ["system", "user", "assistant"]
    assert '"fields"' in messages[-1]["content"]


def test_prepare_sft_and_rl_corpus_from_synthetic_records() -> None:
    sample = SyntheticFormGenerator(seed=9).generate_sample(sample_index=1)
    sft_records = prepare_sft_corpus(
        [
            {
                "task_type": "SFT_gold",
                "input": sample.input.to_dict(),
                "reference": sample.reference.to_dict(),
                "metadata": sample.metadata,
            }
        ]
    )
    rl_records = prepare_rl_corpus(
        [
            {
                "task_type": "RL_prompt_only",
                "input": sample.input.to_dict(),
                "metadata": sample.metadata,
            }
        ],
        reward_profile="rlvr",
    )

    assert sft_records[0]["stage"] == "phase_a_sft"
    assert len(sft_records[0]["messages"]) == 3
    assert rl_records[0]["stage"] == "phase_c_rlvr"
    assert rl_records[0]["reward_profile"] == "rlvr"


def test_prepare_dpo_corpus_uses_preference_structure() -> None:
    preference_record = {
        "sample_id": "sample-pref",
        "input": {
            "sample_id": "sample-pref",
            "form_type": "insurance_application_form",
            "pdf_page": 1,
            "ocr_tokens": [{"text": "投保人姓名", "bbox": [1, 2, 3, 4], "page": 1}],
        },
        "metadata": {"bucket": {"template_family": "template_a"}},
        "reward_profile": "default",
        "reward_margin": 0.2,
        "chosen_candidate_id": "good",
        "rejected_candidate_id": "bad",
        "chosen": {
            "prediction": {
                "sample_id": "sample-pref",
                "fields": {"policyholder_name": "张三"},
                "validations": [],
            }
        },
        "rejected": {
            "prediction": {
                "sample_id": "sample-pref",
                "fields": {"policyholder_name": ""},
                "validations": [],
            }
        },
    }

    corpus = prepare_dpo_corpus([preference_record])

    assert corpus[0]["stage"] == "phase_b_dpo"
    assert corpus[0]["chosen_candidate_id"] == "good"
    assert '"policyholder_name"' in corpus[0]["chosen"]


def test_generate_training_manifests_from_matrix() -> None:
    matrix = load_experiment_matrix(Path("configs/experiment_matrix.yaml"))

    manifests = build_training_manifests(
        matrix,
        train_data_path=Path("outputs/train.phase_b.jsonl"),
        output_dir=Path("outputs/manifests"),
    )

    assert [item.name for item in manifests] == [
        "phase_b_dpo",
        "phase_c_grpo",
        "phase_c_rloo",
    ]
    assert manifests[0].backend == "multi"
    assert manifests[0].runtime["backend_name"] == "trl"
    assert manifests[1].reward_profile == "rlvr"


def test_prepare_training_cli_and_manifest_cli_write_outputs(tmp_path: Path) -> None:
    sample = SyntheticFormGenerator(seed=5).generate_sample(sample_index=2)
    sft_input_path = tmp_path / "sft.jsonl"
    sft_output_path = tmp_path / "sft_corpus.jsonl"
    manifest_dir = tmp_path / "manifests"
    sft_input_path.write_text(
        json.dumps(
            {
                "task_type": "SFT_gold",
                "input": sample.input.to_dict(),
                "reference": sample.reference.to_dict(),
                "metadata": sample.metadata,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = prepare_training_main(
        [
            "--input-path",
            str(sft_input_path),
            "--output-path",
            str(sft_output_path),
            "--stage",
            "phase_a_sft",
        ]
    )

    assert exit_code == 0
    payload = json.loads(sft_output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["stage"] == "phase_a_sft"

    manifest_exit_code = generate_manifests_main(
        [
            "--matrix-path",
            "configs/experiment_matrix.yaml",
            "--train-data-path",
            str(sft_output_path),
            "--output-dir",
            str(manifest_dir),
        ]
    )

    assert manifest_exit_code == 0
    assert (manifest_dir / "phase_b_dpo" / "manifest.json").exists()
    assert (manifest_dir / "phase_c_grpo" / "verl.yaml").exists()
