# AutoDL Runbook

这份 runbook 只保留当前仓库主线，不再区分 online / cached 等旧分支。

## 1. 目录与环境变量

推荐目录：

- 代码、模型、 HF cache：`/root/autodl-fs`
- venv、输出、中间文件：`/root/autodl-tmp/veridoc-rl`

环境变量模板直接使用 [autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example)。

当前建议值：

```bash
export HF_HOME="/root/autodl-fs/.cache/huggingface"
export VERIDOC_PROJECT_ROOT="/root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL"
export VERIDOC_WORK_ROOT="/root/autodl-tmp/veridoc-rl"
export VERIDOC_MODEL_REF="/root/autodl-fs/models/Qwen3-1.7B"
export VERIDOC_MODEL_PATH="${VERIDOC_MODEL_REF}"
export VERIDOC_TRAIN_PYTHON_BIN="${VERIDOC_WORK_ROOT}/.venv-train/bin/python"
export VERIDOC_RL_PYTHON_BIN="${VERIDOC_WORK_ROOT}/.venv-rl/bin/python"
export VERIDOC_OUTPUT_ROOT="${VERIDOC_WORK_ROOT}/pipelines"
export VERIDOC_SFT_GOLD_PATH="${VERIDOC_WORK_ROOT}/outputs/sft_gold.jsonl"
export VERIDOC_RL_PROMPT_ONLY_PATH="${VERIDOC_WORK_ROOT}/outputs/rl_prompt_only.jsonl"
export VERIDOC_API_BASE="http://127.0.0.1:30000/v1"
```

## 2. 安装双环境

仓库内已经锁定两套依赖：

- [autodl.train.txt](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/requirements/autodl.train.txt)
- [autodl.rl.txt](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/requirements/autodl.rl.txt)

执行：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
bash scripts/bootstrap_autodl_envs.sh cu124 all
```

说明：

- `.venv-train` 安装 SFT / DPO / 测试依赖
- `.venv-rl` 安装 SGLang / verl 依赖
- train 与 rl 的 torch 版本故意不同，不要手动合并

## 3. 验证环境

训练环境：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python -c "import torch, transformers, trl; print(torch.__version__, transformers.__version__, trl.__version__)"
deactivate
```

RL 环境：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-rl/bin/activate"
python -c "import torch, pyarrow, verl, sglang; print(torch.__version__, pyarrow.__version__, verl.__version__, sglang.__version__)"
deactivate
```

## 4. 生成数据

`SFT_gold` 和 `RL_prompt_only` 现在都由同一个生成入口构造：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"

python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --output-path "${VERIDOC_SFT_GOLD_PATH}"

python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type RL_prompt_only \
  --output-path "${VERIDOC_RL_PROMPT_ONLY_PATH}"
```

注意：

- 输出字段名为中文
- `SFT_gold` 会校验 ground truth 是否真的来自 OCR 可见内容
- 遇到不可见字段被写入 gold，会直接报错而不是静默导出脏数据

## 5. 启动 SGLang

当前 DPO 默认从 `phase_a_sft` adapter 采样，因此建议统一用支持 LoRA 的启动方式。

基础启动：

```bash
source /tmp/veridoc_autodl.env
ENABLE_LORA=1 \
LORA_PATHS="sft_adapter=${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_a_sft/checkpoints" \
DISABLE_CUDA_GRAPH=1 \
DISABLE_RADIX_CACHE=1 \
bash scripts/start_sglang_server.sh
```

后台启动：

```bash
nohup env \
  ENABLE_LORA=1 \
  LORA_PATHS="sft_adapter=${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_a_sft/checkpoints" \
  DISABLE_CUDA_GRAPH=1 \
  DISABLE_RADIX_CACHE=1 \
  bash scripts/start_sglang_server.sh \
  > "${VERIDOC_WORK_ROOT}/sglang.log" 2>&1 &
```

检查服务：

```bash
curl http://127.0.0.1:30000/v1/models
```

## 6. 跑 Pipeline

prepare-only：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

完整执行：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```

当前默认行为：

- baseline 可选评测保留
- `phase_a_sft` 正常训练
- `phase_b_dpo` 从 SFT adapter 采样候选
- `phase_c_grpo` 正常训练
- 训练后 checkpoint 评测默认关闭

## 7. 单独评测

phase A/B/C 的训练后评测不要放进一键流程。

单独执行：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python scripts/run_phase_a_eval.py \
  --reference-path "${VERIDOC_SFT_GOLD_PATH}" \
  --prediction-path "<your_prediction_jsonl>" \
  --report-path "<your_report_json>"
```

## 8. 排错重点

- 如果输出总带 `<think>`：
  - 当前 pipeline 已默认关闭 thinking
  - 若仍出现，先检查 SGLang 请求侧是否生效，再看 `raw_text`

- 如果 DPO preference 为空：
  - 先确认 `phase_a_sft/checkpoints` 是否存在
  - 再确认 SGLang 是否已挂载 `sft_adapter`
  - 再检查 `phase_b_dpo/candidates.jsonl` 的 reward margin

- 如果 `SFT_gold` 生成失败：
  - 通常是 ground truth 与 OCR 可见内容不一致
  - 这是预期保护，不要跳过
