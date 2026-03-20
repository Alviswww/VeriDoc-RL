# scripts

这份索引只说明仓库脚本入口，以及它们当前各自负责哪一段主链路。

先看现在的默认边界：

- baseline candidate generation 默认走 `SGLang` 的 OpenAI-compatible API
- 仓库主流程默认只需要 `.venv-rl`
- `.venv-rl` 负责开发、测试、`prepare-only`、SFT、DPO、RL
- 当前默认版本岛是 `torch 2.6.0 + sglang 0.4.6.post5 + verl 0.4.1`
- 对 `RTX 2060 / SM75`，SGLang 默认建议加 `--attention-backend triton --sampling-backend pytorch`
- `phase_a_sft` 当前实现是 `transformers.Trainer + datasets + peft`
- `phase_b_dpo` 当前实现是 `TRL DPOTrainer`
- `phase_c_grpo / phase_c_rloo` 当前桥接到 `verl`，默认 rollout backend 是 `sglang`
- `vLLM` 仅保留为可选实验路径，不再是默认环境脚本的一部分

## 数据与评测

- `python scripts/generate_sft_dataset.py`
  - 生成带 bucket 元数据的 synthetic 数据
  - 支持 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
- `python scripts/generate_candidates.py`
  - 读取带 `input` 的 JSONL
  - 通过 OpenAI-compatible API 生成多候选 `candidate.jsonl`
  - 默认推荐接本地 `SGLang`，也兼容 `vLLM`
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

在这之前，先执行：

```bash
bash scripts/rebuild_wsl_envs.sh cu126
```

这个脚本现在默认只重建 `.venv-rl`，也就是本地主链路需要的唯一环境。

如果你后续确实要单独做 `vLLM` 对比，再显式执行：

```bash
bash scripts/rebuild_wsl_envs.sh cu126 --with-vllm
```
