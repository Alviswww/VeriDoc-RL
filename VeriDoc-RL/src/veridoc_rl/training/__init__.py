from __future__ import annotations

from veridoc_rl.training.corpus import (
    export_training_jsonl,
    prepare_dpo_corpus,
    prepare_rl_corpus,
    prepare_sft_corpus,
)
from veridoc_rl.training.finetune import (
    AdapterConfig,
    PrecisionConfig,
    adapter_config_from_mapping,
    load_causal_lm,
    load_tokenizer,
    precision_config_from_mapping,
)
from veridoc_rl.training.manifests import (
    TrainingManifest,
    build_training_manifests,
    render_manifest_markdown,
    render_verl_manifest_yaml,
    write_training_bundle,
)
from veridoc_rl.training.prompting import (
    DEFAULT_SYSTEM_PROMPT,
    build_assistant_response,
    build_chat_messages,
    build_user_prompt,
)
from veridoc_rl.training.runtime import (
    RuntimeLaunchPlan,
    build_runtime_launch_plan,
    execute_runtime_plan,
    load_training_manifest,
    write_runtime_bundle,
)
from veridoc_rl.training.trl_dpo import (
    TrlDPOConfig,
    build_trl_dpo_rows,
    execute_trl_dpo_training,
    export_trl_dpo_dataset,
    load_trl_dpo_config,
    write_trl_dpo_config,
)
from veridoc_rl.training.trl_sft import (
    TrlSFTConfig,
    build_sft_rows,
    execute_trl_sft_training,
    export_sft_dataset,
    load_trl_sft_config,
    write_trl_sft_config,
)
from veridoc_rl.training.verl_reward import compute_score

__all__ = [
    "AdapterConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "PrecisionConfig",
    "RuntimeLaunchPlan",
    "TrlSFTConfig",
    "TrainingManifest",
    "TrlDPOConfig",
    "adapter_config_from_mapping",
    "build_assistant_response",
    "build_chat_messages",
    "build_runtime_launch_plan",
    "build_sft_rows",
    "build_trl_dpo_rows",
    "build_training_manifests",
    "build_user_prompt",
    "compute_score",
    "execute_trl_sft_training",
    "execute_runtime_plan",
    "execute_trl_dpo_training",
    "export_training_jsonl",
    "export_sft_dataset",
    "export_trl_dpo_dataset",
    "load_causal_lm",
    "load_training_manifest",
    "load_tokenizer",
    "load_trl_sft_config",
    "load_trl_dpo_config",
    "precision_config_from_mapping",
    "prepare_dpo_corpus",
    "prepare_rl_corpus",
    "prepare_sft_corpus",
    "render_manifest_markdown",
    "render_verl_manifest_yaml",
    "write_runtime_bundle",
    "write_training_bundle",
    "write_trl_sft_config",
    "write_trl_dpo_config",
]
