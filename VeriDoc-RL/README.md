# VeriDoc-RL

VeriDoc-RL 是一个面向投保单 OCR 结构化抽取的后训练仓库。当前仓库已经收敛到单一主线：

- 输出协议使用中文字段名与中文规则 `rule_id`
- `SFT_gold` 只基于 OCR 可见内容构造，不再编造不可见字段
- `DPO` 默认从 `phase_a_sft` 的 QLoRA adapter 采样候选
- Qwen3 默认关闭 thinking 输出，并在解析端兜底清洗 `<think>...</think>`
- 训练后评测默认不纳入一键 pipeline，需要单独执行

## 保留文档

- 执行手册：[autodl_runbook.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_runbook.md)
- 流水线说明：[pipeline_deep_dive.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/pipeline_deep_dive.md)

## 关键入口

- 主配置：[pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml)
- 环境模板：[autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example)
- 环境安装：[bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh)
- 启动 SGLang：[start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh)
- 生成数据：[generate_sft_dataset.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/generate_sft_dataset.py)
- 运行 pipeline：[run_pipeline.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/run_pipeline.py)

## 快速开始

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
cp configs/autodl.env.example /tmp/veridoc_autodl.env
source /tmp/veridoc_autodl.env

bash scripts/bootstrap_autodl_envs.sh cu124 all

ENABLE_LORA=1 \
LORA_PATHS="sft_adapter=${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_a_sft/checkpoints" \
DISABLE_CUDA_GRAPH=1 \
DISABLE_RADIX_CACHE=1 \
bash scripts/start_sglang_server.sh
```

然后：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python scripts/generate_sft_dataset.py --count 200 --seed 7 --output-path "${VERIDOC_SFT_GOLD_PATH}"
python scripts/generate_sft_dataset.py --count 200 --seed 7 --task-type RL_prompt_only --output-path "${VERIDOC_RL_PROMPT_ONLY_PATH}"
python scripts/run_pipeline.py --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```

## 目录说明

- `src/veridoc_rl/`: 核心实现
- `scripts/`: CLI 入口
- `configs/`: AutoDL 与 pipeline 配置
- `requirements/`: train / rl 双环境锁定依赖
- `docs/`: 保留执行与原理文档

## 当前约束

- 推荐运行环境是 AutoDL 单机 GPU。
- `pytest`、SFT、DPO 用 `.venv-train`。
- `SGLang`、`verl` 用 `.venv-rl`。
- 若要做 phase A/B/C checkpoint 评测，请单独运行评测脚本，不要依赖 pipeline 自动执行。
