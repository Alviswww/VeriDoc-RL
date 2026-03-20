# VeriDoc-RL

VeriDoc-RL 是一个面向制式投保单结构化抽取的 verifier-guided post-training 项目。它把 `OCR token / 表单文本 -> 结构化字段抽取 -> 规则校验 -> verifier reward` 串成一条统一链路，用同一套 schema、规则、verifier 和 reward 支持 `baseline / SFT / DPO / RLVR` 对比。

当前仓库的默认运行路线已经切到 **云端单机 GPU**，优先面向 **AutoDL** 这类租赁平台，而不是继续围绕本地 WSL 老卡调环境。

当前主链路的推理后端也固定收敛到 **SGLang**：

- baseline 多候选采样走 `SGLang`
- `phase_c_*` rollout 走 `verl + SGLang`
- `vLLM` 只保留为后续吞吐对比的可选路径，不再是默认前置条件

## 1. 当前默认主线

当前主线固定为：

1. baseline：`models/Qwen3-0.6B`
2. `phase_a_sft`：在 baseline 上做 SFT
3. `phase_b_dpo`：在 `phase_a_sft` checkpoint 上做 DPO
4. `phase_c_grpo` 或 `phase_c_rloo`：在 `phase_a_sft` checkpoint 上做 RLVR
5. 对比 `baseline / sft / dpo / rlvr`

这里保留两个刻意固定的约束：

- `DPO` 默认接 `SFT checkpoint`
- `RLVR` 默认也接 `SFT checkpoint`

这样可以把 “先让模型稳定输出 JSON” 和 “偏好优化 / verifier reward 优化” 分开。

## 2. 框架边界

| 环节 | 当前框架 | 说明 |
| --- | --- | --- |
| baseline candidate generation | `SGLang` | 通过 OpenAI-compatible API 生成多候选 |
| checkpoint 回评推理 | `transformers` + `peft` | `scripts/run_inference.py` 直接加载本地模型或 checkpoint |
| `phase_a_sft` | `transformers.Trainer` + `datasets` + `peft` | 实现在 `src/veridoc_rl/training/trl_sft.py` |
| `phase_b_dpo` | `TRL DPOTrainer` | 实现在 `src/veridoc_rl/training/trl_dpo.py` |
| `phase_c_grpo / phase_c_rloo` | `verl` | 通过 `src/veridoc_rl/training/verl_reward.py` 接 verifier reward |

结论：

- baseline 默认走 `SGLang`
- SFT 和 checkpoint inference 默认走本地 `transformers`
- DPO 默认走 `TRL`
- RL 默认走 `verl`
- 当前不需要独立 reward model

## 3. 为什么默认切到云端

本项目当前的主要环境冲突点，不是代码逻辑，而是训练栈和服务栈的 Python 依赖组合：

- `sglang[srt]` 更适合作为单独的 serving / RL 环境
- `trl + transformers + peft` 更适合作为单独的训练环境
- 把它们强行塞进一个 `.venv`，在本地老卡上调试成本很高

所以仓库现在默认采用 **双环境**：

- `.venv-train`
  - 用于开发、测试、SFT、DPO、checkpoint inference
- `.venv-rl`
  - 用于 `SGLang` 启动与 `verl` rollout

这不代表上云后就完全没有依赖冲突，只是把问题从“本地老卡很难调”变成了“云端更容易维护兼容岛”：

- 训练栈和 serving / rollout 栈仍然要分开
- `torch / sglang / verl / flashinfer` 仍然要一起 pin
- 但 AutoDL 至少能先把 GPU、CUDA 和磁盘布局稳定下来

## 4. AutoDL 目录约定

推荐把代码和模型、输出目录分开：

- 持久化代码与模型：`/root/autodl-fs`
- 高 IO 输出与中间产物：`/root/autodl-tmp`

仓库已经附带 AutoDL 环境变量模板：

```bash
source configs/autodl.env.example
```

你通常会按下面这种方式调整：

```bash
export VERIDOC_PROJECT_ROOT="/root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL"
export VERIDOC_WORK_ROOT="/root/autodl-tmp/veridoc-rl"
export VERIDOC_MODEL_PATH="${VERIDOC_PROJECT_ROOT}/models/Qwen3-0.6B"
export VERIDOC_TRAIN_PYTHON_BIN="${VERIDOC_PROJECT_ROOT}/.venv-train/bin/python"
export VERIDOC_RL_PYTHON_BIN="${VERIDOC_PROJECT_ROOT}/.venv-rl/bin/python"
export VERIDOC_OUTPUT_ROOT="${VERIDOC_WORK_ROOT}/pipelines"
export VERIDOC_SFT_GOLD_PATH="${VERIDOC_WORK_ROOT}/outputs/sft_gold.jsonl"
export VERIDOC_RL_PROMPT_ONLY_PATH="${VERIDOC_WORK_ROOT}/outputs/rl_prompt_only.jsonl"
export VERIDOC_API_BASE="http://127.0.0.1:30000/v1"
```

新增的 [pipeline.autodl.qwen3_0p6.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_0p6.yaml) 会自动读取这些环境变量。

## 5. 快速开始

### 5.1 准备实例

建议直接选 Ubuntu 容器实例，并确保：

- GPU 在容器内可见
- `nvidia-smi` 正常
- Python 3.12 可用

如果镜像里没有 `python3.12`，先在实例内准备一个 Python 3.12。

### 5.2 克隆仓库

```bash
mkdir -p /root/autodl-fs/code
cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd VeriDoc-RL/VeriDoc-RL
```

### 5.3 重建环境

默认入口改为：

```bash
bash scripts/bootstrap_autodl_envs.sh auto
```

如果你已经激活了自己的 Python 3.12 环境，也可以：

```bash
PYTHON_BIN=python bash scripts/bootstrap_autodl_envs.sh auto
```

这个脚本会：

- 自动检测 `cu126` 或 `cu124`
- 重建 `.venv-train`
- 重建 `.venv-rl`
- 把训练依赖和 `sglang/verl` 依赖拆开

### 5.4 准备工作目录

```bash
mkdir -p /root/autodl-tmp/veridoc-rl/outputs
mkdir -p /root/autodl-tmp/veridoc-rl/pipelines
source configs/autodl.env.example
```

根据你的实际仓库路径，先改好 [autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example) 再 `source`。

### 5.5 验证两套环境

```bash
source .venv-train/bin/activate
python -c "import torch, transformers, trl; print(torch.cuda.is_available(), torch.__version__, transformers.__version__, trl.__version__)"

source .venv-rl/bin/activate
python -c "import torch, pyarrow, verl, sglang, fastapi, uvicorn; print(torch.cuda.is_available(), torch.__version__, verl.__version__, sglang.__version__, uvicorn.__version__)"
```

## 6. 第一次跑通项目的推荐顺序

### 6.1 先跑测试

```bash
source .venv-train/bin/activate
pytest
veridoc-rl-smoke
```

### 6.2 生成数据

```bash
source .venv-train/bin/activate

python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path "${VERIDOC_SFT_GOLD_PATH}"

python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 11 \
  --task-type RL_prompt_only \
  --output-path "${VERIDOC_RL_PROMPT_ONLY_PATH}"
```

### 6.3 启动 SGLang

```bash
bash scripts/start_sglang_server.sh
```

脚本现在会优先读取 `VERIDOC_MODEL_PATH`。如果你想临时覆盖，传 `MODEL_PATH` 即可：

```bash
MODEL_PATH="/root/autodl-fs/models/Qwen3-0.6B" bash scripts/start_sglang_server.sh
```

如果你需要额外参数，可以直接追加到脚本后面：

```bash
bash scripts/start_sglang_server.sh --trust-remote-code
```

另开一个终端验证：

```bash
curl http://127.0.0.1:30000/v1/models
```

### 6.4 生成 candidates

```bash
source .venv-train/bin/activate

python scripts/generate_candidates.py \
  --input-path "${VERIDOC_SFT_GOLD_PATH}" \
  --output-path "${VERIDOC_WORK_ROOT}/outputs/candidates.jsonl" \
  --model "${VERIDOC_MODEL_PATH}" \
  --api-base "${VERIDOC_API_BASE}" \
  --num-candidates 4 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 1024
```

### 6.5 生成 preferences

```bash
source .venv-train/bin/activate

python scripts/generate_preference_dataset.py \
  --reference-path "${VERIDOC_SFT_GOLD_PATH}" \
  --candidate-path "${VERIDOC_WORK_ROOT}/outputs/candidates.jsonl" \
  --output-path "${VERIDOC_WORK_ROOT}/outputs/preferences.jsonl" \
  --min-margin 0.05
```

### 6.6 跑 pipeline

仓库默认保留一个相对路径配置：

- [pipeline.qwen3_0p6.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.qwen3_0p6.yaml)

云端默认新增一个 AutoDL 配置：

- [pipeline.autodl.qwen3_0p6.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_0p6.yaml)

先做 prepare-only：

```bash
source .venv-train/bin/activate
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_0p6.yaml \
  --prepare-only
```

再正式执行：

```bash
source .venv-train/bin/activate
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_0p6.yaml
```

现在整条 pipeline 可以从 `.venv-train` 发起：

- `phase_a_sft / phase_b_dpo` 默认走 `VERIDOC_TRAIN_PYTHON_BIN`
- `phase_c_grpo / phase_c_rloo` 默认自动切到 `VERIDOC_RL_PYTHON_BIN`

## 7. 当前仓库里和云端相关的改动

这次改造新增了：

- [bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh)
  - AutoDL 双环境重建脚本
- [start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh)
  - 用 `.venv-rl` 启动 `SGLang`
- [autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example)
  - AutoDL 环境变量模板
- [pipeline.autodl.qwen3_0p6.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_0p6.yaml)
  - AutoDL pipeline spec

同时，pipeline spec 和 training manifest 现在支持环境变量展开，所以你可以在 YAML / JSON 里直接写 `${VERIDOC_OUTPUT_ROOT}` 这类路径。

## 8. 旧的本地 WSL 脚本

仓库里仍然保留：

- [rebuild_wsl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/rebuild_wsl_envs.sh)

它现在属于 **兼容旧场景的本地脚本**，不再是默认入口。
如果你之后仍要回到本地 WSL，先看这个脚本，再看 runbook；但默认文档和默认执行路线已经改成云端。

## 9. 下一步建议

如果你的目标是先在 AutoDL 上把项目完整跑通，建议按这个节奏：

1. 先跑 `generate_sft_dataset.py`
2. 再确认 `SGLang` 能在 `.venv-rl` 里稳定启动
3. 先用 `--prepare-only` 跑通 pipeline 产物
4. 最后再放开 `SFT / DPO / RL`

更细的操作顺序看 [local_data_cloud_training_runbook.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/local_data_cloud_training_runbook.md)。
