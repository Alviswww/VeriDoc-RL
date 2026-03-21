# AutoDL 方案 A：直接使用 Hugging Face 模型

这是当前推荐方案，目标是最快把 `baseline -> SFT -> DPO -> RL` 全链路跑通。

适用场景：

- 你现在主要目标是先跑通仓库
- 机器可以稳定联网 Hugging Face
- 你不想先手工下载模型快照

## 1. 机器与目录准备

建议实例条件：

- GPU：RTX 4090 或 5090
- 系统：Ubuntu 容器
- CUDA：12.4 或 12.6
- Python：3.12

建议目录：

- 持久化代码与缓存：`/root/autodl-fs`
- 中间产物与输出：`/root/autodl-tmp`

```bash
mkdir -p /root/autodl-fs/code
mkdir -p /root/autodl-tmp/veridoc-rl/outputs
mkdir -p /root/autodl-tmp/veridoc-rl/pipelines
```

## 2. 克隆仓库

```bash
cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd VeriDoc-RL/VeriDoc-RL
```

## 3. 准备环境变量

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
export VERIDOC_OUTPUT_ROOT="${VERIDOC_WORK_ROOT}/pipelines"
export VERIDOC_SFT_GOLD_PATH="${VERIDOC_WORK_ROOT}/outputs/sft_gold.jsonl"
export VERIDOC_RL_PROMPT_ONLY_PATH="${VERIDOC_WORK_ROOT}/outputs/rl_prompt_only.jsonl"
export VERIDOC_API_BASE="http://127.0.0.1:30000/v1"
```

然后加载：

```bash
source /tmp/veridoc_autodl.env
```

## 4. 重建双环境

```bash
bash scripts/bootstrap_autodl_envs.sh auto all
```

如果你的 Python 3.12 不在默认 PATH：

```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/bootstrap_autodl_envs.sh auto all
```

说明：

- `all` 会同时重建 `${VERIDOC_WORK_ROOT}/.venv-train` 和 `${VERIDOC_WORK_ROOT}/.venv-rl`
- 只想先排查训练栈时，可以用 `auto train`
- 只想单独排查 `SGLang + verl`，可以用 `auto rl`
- 默认会把 venv 和 pip/uv cache 放到 `VERIDOC_WORK_ROOT` 下，避免把系统盘快速写满
- 默认允许在无卡状态下完成安装；如果你想强制要求 GPU 已就绪，再加 `ALLOW_NO_GPU=0`
- `FLASHINFER_TORCH_SERIES` 会跟随 `TORCH_VERSION` 自动推导；如果你手动改 `TORCH_VERSION`，一般不需要再单独改它

## 5. 验证安装

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python -c "import torch, transformers, trl; print(torch.cuda.is_available(), torch.__version__, transformers.__version__, trl.__version__)"
pytest
veridoc-rl-smoke
```

```bash
source "${VERIDOC_WORK_ROOT}/.venv-rl/bin/activate"
python -c "import torch, pyarrow, verl, sglang; print(torch.cuda.is_available(), torch.__version__, verl.__version__, sglang.__version__)"
```

## 6. 准备数据

```bash
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
```

## 7. 启动 SGLang

直接使用 HF repo id：

```bash
bash scripts/start_sglang_server.sh
```

如果你需要显式传参：

```bash
MODEL_REF="Qwen/Qwen3-1.7B" \
bash scripts/start_sglang_server.sh --trust-remote-code
```

验证服务：

```bash
curl http://127.0.0.1:30000/v1/models
```

## 8. 先跑 prepare-only

```bash
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

这里通了，再正式执行：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```

## 9. 常见问题

- `KeyError: 'qwen3'`
  - 说明 `transformers` 太旧。仓库已经把下限提到 `4.51+`，请重新执行环境脚本。
- `SGLang` 安装阶段失败
  - 先确认实例 CUDA 是 12.4 或 12.6。
  - 无卡安装阶段只需要 `/usr/local/cuda/version.json` 或显式传 `cu124 / cu126`；真正开卡运行前再确认 `nvidia-smi` 正常。
- 第一次启动 `SGLang` 比较慢
  - 这是正常现象，因为它会先把模型缓存到 `HF_HOME`。
