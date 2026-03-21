# AutoDL 详细执行 Runbook

这份 runbook 面向第一次在 AutoDL 上跑这个仓库的人。

目标不是先读懂全部源码，而是先把下面这条主线跑通：

1. 在无卡状态下完成代码、环境变量、双环境安装。
2. 开卡后验证 CUDA / SGLang / API 服务。
3. 生成数据。
4. 先跑 `prepare-only`。
5. 再跑完整 pipeline。

## 0. 你只需要先认识这些文件

| 文件 | 用途 |
|---|---|
| [configs/autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example) | AutoDL 环境变量模板，定义项目根目录、工作目录、模型位置、输出目录。 |
| [scripts/bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh) | 创建 `.venv-train` 和 `.venv-rl`，安装训练与 RL/SGLang 依赖。 |
| [scripts/start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh) | 用 RL 环境启动本地 `SGLang` 服务。 |
| [configs/pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml) | AutoDL 主线 pipeline 配置。 |
| [configs/experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml) | SFT / DPO / RL 的训练矩阵配置。 |
| [scripts/generate_sft_dataset.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/generate_sft_dataset.py) | 生成 `SFT_gold` 和 `RL_prompt_only` 数据。 |
| [scripts/run_pipeline.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/run_pipeline.py) | 读取 pipeline spec，串起 baseline / SFT / DPO / RL 全流程。 |

你可以先把仓库理解成两部分：

- `scripts/`：你实际手动执行的入口。
- `src/veridoc_rl/`：这些入口背后的实现。

## 1. 推荐目录布局

在 AutoDL 上，建议这样分工：

- 代码仓库、模型、HF 缓存：`/root/autodl-fs`
- 双环境、pip/uv cache、输出、中间文件：`/root/autodl-tmp`

先创建目录：

```bash
mkdir -p /root/autodl-fs/code
mkdir -p /root/autodl-fs/models
mkdir -p /root/autodl-fs/.cache/huggingface
mkdir -p /root/autodl-tmp/veridoc-rl/outputs
mkdir -p /root/autodl-tmp/veridoc-rl/pipelines
```

推荐把仓库放到这里：

```bash
cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
```

## 2. 先准备环境变量

复制模板：

```bash
cp configs/autodl.env.example /tmp/veridoc_autodl.env
```

把 `/tmp/veridoc_autodl.env` 至少改成下面这些值：

```bash
export HF_HOME="/root/autodl-fs/.cache/huggingface"
export VERIDOC_PROJECT_ROOT="/root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL"
export VERIDOC_WORK_ROOT="/root/autodl-tmp/veridoc-rl"
export VERIDOC_MODEL_REF="Qwen/Qwen3-1.7B"
export VERIDOC_MODEL_PATH="${VERIDOC_MODEL_REF}"
export VERIDOC_TRAIN_PYTHON_BIN="${VERIDOC_WORK_ROOT}/.venv-train/bin/python"
export VERIDOC_RL_PYTHON_BIN="${VERIDOC_WORK_ROOT}/.venv-rl/bin/python"
export VERIDOC_OUTPUT_ROOT="${VERIDOC_WORK_ROOT}/pipelines"
export VERIDOC_SFT_GOLD_PATH="${VERIDOC_WORK_ROOT}/outputs/sft_gold.jsonl"
export VERIDOC_RL_PROMPT_ONLY_PATH="${VERIDOC_WORK_ROOT}/outputs/rl_prompt_only.jsonl"
export VERIDOC_API_BASE="http://127.0.0.1:30000/v1"
```

加载环境变量：

```bash
source /tmp/veridoc_autodl.env
```

确认变量生效：

```bash
echo "$VERIDOC_PROJECT_ROOT"
echo "$VERIDOC_WORK_ROOT"
echo "$VERIDOC_MODEL_REF"
```

## 3. 无卡阶段安装双环境

当前 `bootstrap_autodl_envs.sh` 已支持无卡安装。

默认行为：

- 允许 `nvidia-smi` 不可用。
- 默认把 venv 和 cache 放到 `VERIDOC_WORK_ROOT`。
- `FLASHINFER_TORCH_SERIES` 会自动按 `TORCH_VERSION` 推导。

### 3.1 方案 A：直接使用仓库当前默认依赖

当前仓库默认会在 venv 内安装：

- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`

注意，这和你选择的 AutoDL 基础镜像里“预装了什么 torch”是两层概念：

- 基础镜像自带的是系统环境。
- 这个仓库实际运行时使用的是脚本创建的独立 venv。

如果你接受仓库当前默认依赖，直接执行：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
bash scripts/bootstrap_autodl_envs.sh auto all
```

如果无卡状态下 `auto` 无法识别 CUDA 版本，就显式指定：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
bash scripts/bootstrap_autodl_envs.sh cu124 all
```

### 3.2 方案 B：让 venv 也对齐到 `torch 2.5.1 + cuda 12.4`

如果你希望 venv 内依赖与所选 AutoDL 镜像尽量一致，可以显式覆盖版本。

根据 PyTorch 官方 previous versions 页面，`torch 2.5.1` 对应的 wheel 组合是：

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`

执行命令：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env

TORCH_VERSION=2.5.1 \
TORCHVISION_VERSION=0.20.1 \
TORCHAUDIO_VERSION=2.5.1 \
bash scripts/bootstrap_autodl_envs.sh cu124 all
```

这时脚本会自动把 `FLASHINFER_TORCH_SERIES` 推导成 `torch2.5`，不需要再单独手动改。

### 3.3 安装后检查

训练环境：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python -V
python -c "import torch, transformers, trl; print(torch.__version__, transformers.__version__, trl.__version__)"
pytest
veridoc-rl-smoke
deactivate
```

RL 环境：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-rl/bin/activate"
python -V
python -c "import torch, pyarrow, verl, sglang; print(torch.__version__, pyarrow.__version__, verl.__version__, sglang.__version__)"
deactivate
```

如果这里已经通过，说明“无卡安装阶段”完成。

## 4. 开卡后的第一轮检查

开卡后先不要急着跑 pipeline，先确认 GPU 运行时真的可用：

```bash
nvidia-smi
```

确认 CUDA 目录：

```bash
ls -la /usr/local/cuda
cat /usr/local/cuda/version.json
```

再验证两个环境都能识别 GPU：

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
deactivate
```

```bash
source "${VERIDOC_WORK_ROOT}/.venv-rl/bin/activate"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
deactivate
```

如果这里 `torch.cuda.is_available()` 仍然是 `False`，不要继续往下跑，先排查镜像、CUDA、开卡状态。

## 5. 启动 SGLang 服务

先进入仓库并加载环境变量：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
```

前台启动：

```bash
bash scripts/start_sglang_server.sh
```

如果你想让它在后台跑：

```bash
nohup bash scripts/start_sglang_server.sh > "${VERIDOC_WORK_ROOT}/sglang.log" 2>&1 &
```

检查服务是否起来：

```bash
curl http://127.0.0.1:30000/v1/models
```

如果 `VERIDOC_MODEL_REF` 指向 HF repo id，第一次启动比较慢，因为会先把模型缓存到 `HF_HOME`。

如果你想先把模型缓存到本地目录，再启动服务，参考：

- [autodl_cached_model_setup.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_cached_model_setup.md)
- [scripts/prefetch_hf_model.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/prefetch_hf_model.py)

## 6. 生成数据

使用 train 环境：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
```

生成 `SFT_gold`：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path "${VERIDOC_SFT_GOLD_PATH}"
```

生成 `RL_prompt_only`：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 11 \
  --task-type RL_prompt_only \
  --output-path "${VERIDOC_RL_PROMPT_ONLY_PATH}"
```

确认文件已经生成：

```bash
ls -lh "${VERIDOC_SFT_GOLD_PATH}"
ls -lh "${VERIDOC_RL_PROMPT_ONLY_PATH}"
```

## 7. 先跑 prepare-only

这一步非常重要。它会先把训练数据、manifest、runtime plan 都准备出来，但不真正执行训练。

执行：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

执行完后，重点看这个目录：

```bash
ls -la "${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl"
```

这里应该能看到：

- `state.json`
- `spec.snapshot.json`
- `baseline/`
- `phase_a_sft/`
- `phase_b_dpo/`
- `phase_c_grpo/`

每个训练阶段目录下，通常至少会出现：

- `train.jsonl`
- `manifest.json`
- `runtime_plan.json`
- `launch.sh`

你可以手动检查其中一个阶段：

```bash
cat "${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_a_sft/runtime_plan.json"
```

如果 `prepare-only` 都没通过，不要直接进入完整训练。

## 8. 再跑完整 pipeline

确认：

- GPU 已开。
- `SGLang` 已正常服务。
- 两份数据已生成。
- `prepare-only` 已通过。

然后正式执行：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```

如果你希望忽略旧状态重新跑：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --no-resume
```

## 9. 结果文件在哪里

当前 AutoDL spec 的 `run.name` 是 `qwen3_1p7_autodl`，所以主输出目录通常是：

```text
/root/autodl-tmp/veridoc-rl/pipelines/qwen3_1p7_autodl
```

重点文件：

- `state.json`
  - 当前 pipeline 执行状态。
- `summary.json`
  - 最终汇总。
- `comparison/`
  - 多阶段评测对比产物。
- `baseline/report.json`
  - baseline 评测结果。
- `phase_a_sft/report.json`
  - SFT checkpoint 评测结果。
- `phase_b_dpo/report.json`
  - DPO checkpoint 评测结果。
- `phase_c_grpo/report.json`
  - RL checkpoint 评测结果。

## 10. 什么时候看哪个脚本

如果你遇到问题，按这个顺序找：

1. 环境没装起来。
   看 [scripts/bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh)
2. SGLang 起不来。
   看 [scripts/start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh)
3. 数据文件没生成。
   看 [scripts/generate_sft_dataset.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/generate_sft_dataset.py)
4. pipeline 停在某个阶段。
   先看输出目录下该阶段的 `runtime_plan.json`、`launch.sh`、`manifest.json`
5. 想知道 pipeline 到底做了什么。
   看 [src/veridoc_rl/orchestration/stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py)
6. 想知道 spec 各字段是什么意思。
   看 [src/veridoc_rl/orchestration/spec.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/spec.py)

## 11. 最常见的失败点

### 11.1 无卡安装时 `auto` 识别不到 CUDA

原因：

- 没有 GPU。
- `nvidia-smi` 不可用。
- `/usr/local/cuda/version.json` 也不可用。

处理：

```bash
bash scripts/bootstrap_autodl_envs.sh cu124 all
```

### 11.2 仓库放错目录导致系统盘爆掉

处理原则：

- 仓库放 `/root/autodl-fs/code`
- `VERIDOC_WORK_ROOT` 放 `/root/autodl-tmp/veridoc-rl`

不要把双环境默认留在系统盘小目录里。

### 11.3 SGLang 启动失败

先检查：

```bash
nvidia-smi
curl http://127.0.0.1:30000/v1/models
tail -n 200 "${VERIDOC_WORK_ROOT}/sglang.log"
```

再确认：

- `VERIDOC_MODEL_REF` 指向的 repo id 或本地目录是对的。
- `VERIDOC_RL_PYTHON_BIN` 指向 `${VERIDOC_WORK_ROOT}/.venv-rl/bin/python`。

### 11.4 完整 pipeline 卡在 RL 阶段

先看：

```bash
cat "${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_c_grpo/runtime_plan.json"
cat "${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl/phase_c_grpo/launch.sh"
```

通常优先排查：

- `SGLang` 服务是否还活着。
- `VERIDOC_API_BASE` 是否还是 `http://127.0.0.1:30000/v1`。
- RL 环境是否真的能看到 GPU。

## 12. 最短执行清单

如果你只想照抄，不想来回翻文档，就按这个顺序执行。

无卡阶段：

```bash
mkdir -p /root/autodl-fs/code
mkdir -p /root/autodl-tmp/veridoc-rl/outputs
mkdir -p /root/autodl-tmp/veridoc-rl/pipelines

cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL

cp configs/autodl.env.example /tmp/veridoc_autodl.env
source /tmp/veridoc_autodl.env

bash scripts/bootstrap_autodl_envs.sh cu124 all

source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
pytest
veridoc-rl-smoke
deactivate
```

开卡后：

```bash
cd /root/autodl-fs/code/VeriDoc-RL/VeriDoc-RL
source /tmp/veridoc_autodl.env

nvidia-smi

nohup bash scripts/start_sglang_server.sh > "${VERIDOC_WORK_ROOT}/sglang.log" 2>&1 &
curl http://127.0.0.1:30000/v1/models

source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"

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

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```
