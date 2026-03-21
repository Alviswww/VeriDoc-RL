# scripts

这份索引只保留 **AutoDL 云端主线**。

## 环境与服务

- `bash scripts/bootstrap_autodl_envs.sh`
  - 默认重建两套环境
  - `.venv-train`：SFT / DPO / inference / tests
  - `.venv-rl`：`SGLang` serving + `verl`
  - 支持第二个参数 `all | train | rl`
- `bash scripts/start_sglang_server.sh`
  - 使用 `.venv-rl` 启动本地 `SGLang`
  - 默认优先读取 `MODEL_REF / VERIDOC_MODEL_REF`
  - 同时兼容旧变量 `MODEL_PATH / VERIDOC_MODEL_PATH`
  - 支持 Hugging Face repo id 和本地模型目录
  - 支持 `HOST / PORT / ATTENTION_BACKEND / SAMPLING_BACKEND` 环境变量覆盖
  - 额外的 SGLang 参数可以直接追加在脚本命令后面
- `python scripts/prefetch_hf_model.py`
  - 把 `Qwen/Qwen3-1.7B` 或其他 HF 模型快照下载到本地目录
  - 适合做 AutoDL 本地缓存方案

## 数据与评测

- `python scripts/generate_sft_dataset.py`
  - 生成 synthetic 数据
  - 支持 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
- `python scripts/generate_candidates.py`
  - 通过 OpenAI-compatible API 生成多候选
  - 默认推荐接 `SGLang`
- `python scripts/generate_preference_dataset.py`
  - 用 verifier + composite reward 生成 DPO preference
- `python scripts/run_phase_a_eval.py`
  - 运行 verifier suite 并输出报告
- `python scripts/compare_phase_reports.py`
  - 对比多份评测报告
- `python scripts/generate_experiment_plan.py`
  - 从 experiment matrix 导出实验计划

## 训练准备

- `python scripts/prepare_training_data.py`
  - 生成 `phase_a_sft / phase_b_dpo / phase_c_rlvr` 训练语料
- `python scripts/generate_training_manifests.py`
  - 生成 phase 级 manifest bundle
- `python scripts/prepare_training_runtime.py`
  - 生成 `runtime_plan.json` 与 `launch.sh`
- `python scripts/prepare_verl_runtime.py`
  - `prepare_training_runtime.py` 的兼容别名

## 推理与编排

- `python scripts/run_inference.py`
  - 用本地 `transformers` / checkpoint 做离线推理
- `python scripts/run_pipeline.py`
  - 读取单个 pipeline spec
  - 串起 baseline / SFT / DPO / RLVR
  - 维护 `state.json`、`summary.json`
  - 默认从 `.venv-train` 发起，但 RL 阶段会自动切到 `VERIDOC_RL_PYTHON_BIN`

## 云端推荐顺序

1. `bash scripts/bootstrap_autodl_envs.sh auto all`
2. `source configs/autodl.env.example`
3. `pytest`
4. `python scripts/generate_sft_dataset.py`
5. `bash scripts/start_sglang_server.sh`
6. `python scripts/run_pipeline.py --spec-path configs/pipeline.autodl.qwen3_1p7.yaml --prepare-only`
7. `python scripts/run_pipeline.py --spec-path configs/pipeline.autodl.qwen3_1p7.yaml`
