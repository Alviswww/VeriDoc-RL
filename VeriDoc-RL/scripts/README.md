# scripts

这份索引专门说明仓库脚本入口，以及它们各自对应的框架。

先看最关键的边界：

- `vLLM` 只负责 baseline candidate generation
- `vLLM` 推荐放在单独的 `.venv-vllm`
- `transformers + peft` 负责 checkpoint 本地推理
- 仓库主流程推荐放在 `.venv-rl`
- `phase_a_sft` 负责本地 SFT，当前实现是 `transformers.Trainer + datasets + peft`
- `phase_b_dpo` 负责本地 DPO，当前实现是 `TRL DPOTrainer`
- `phase_c_grpo / phase_c_rloo` 负责 RL，当前桥接到 `verl`，默认 rollout backend 是 `sglang`

## 数据与评测

- `python scripts/generate_sft_dataset.py`
  - 生成带 bucket 元数据的 synthetic 数据
  - 支持 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
- `python scripts/generate_candidates.py`
  - 读取带 `input` 的 JSONL
  - 通过 `vLLM` OpenAI-compatible API 生成多候选 `candidate.jsonl`
  - 输出格式可直接喂给 `generate_preference_dataset.py`
- `python scripts/generate_preference_dataset.py`
  - 读取 reference 和 candidate JSONL
  - 用 verifier + composite reward 生成 `DPO_preference`
- `python scripts/run_phase_a_eval.py`
  - 读取 reference / prediction JSONL
  - 运行 verifier suite
  - 输出报告与 cases
- `python scripts/compare_phase_reports.py`
  - 读取多份评测报告
  - 输出对比报告和 SVG 图
- `python scripts/generate_experiment_plan.py`
  - 读取 `configs/experiment_matrix.yaml`
  - 导出实验计划 JSON / Markdown

## 训练准备

- `python scripts/prepare_training_data.py`
  - 把 `SFT_gold` / `DPO_preference` / `RL_prompt_only` 转成训练语料
  - 输出 `phase_a_sft` / `phase_b_dpo` / `phase_c_rlvr` 可消费 JSONL
- `python scripts/generate_training_manifests.py`
  - 读取 `configs/experiment_matrix.yaml`
  - 按 phase 生成 manifest bundle
  - 支持分别传入 train data 与 base model
- `python scripts/prepare_training_runtime.py`
  - 读取单个 `manifest.json`
  - 生成 `runtime_plan.json` 与 `launch.sh`
  - `phase_a_sft` 会桥接到仓库内 SFT runner
  - `phase_b_dpo` 会桥接到仓库内 TRL DPO runner
  - `phase_c_grpo / phase_c_rloo` 会桥接到 `verl.trainer.main_ppo`
- `python scripts/prepare_verl_runtime.py`
  - `prepare_training_runtime.py` 的兼容别名

## 推理与编排

- `python scripts/run_inference.py`
  - 读取带 `input` 的 JSONL
  - 使用本地 `transformers` / checkpoint 做离线推理
  - 输出统一 `predictions.jsonl`
- `python scripts/run_pipeline.py`
  - 读取单个 pipeline spec
  - 串起 baseline / SFT / DPO / RLVR
  - 维护 `state.json`、`summary.json` 和 checkpoint 依赖

## 推荐启动顺序

如果你第一次运行项目，推荐顺序：

1. `generate_sft_dataset.py`
2. `generate_candidates.py`
3. `generate_preference_dataset.py`
4. `prepare_training_data.py`
5. `generate_training_manifests.py`
6. `prepare_training_runtime.py`
7. `run_pipeline.py --prepare-only`
8. `run_pipeline.py`
