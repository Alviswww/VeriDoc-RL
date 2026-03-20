# VeriDoc-RL AutoDL 执行清单

这份 runbook 现在改成 **AutoDL 云端单机执行清单**。默认假设是：

- 你在 AutoDL 上租了一台 GPU 实例
- 代码和模型放在 `/root/autodl-fs`
- 高 IO 中间产物、训练输出、pipeline 目录放在 `/root/autodl-tmp`
- 训练环境和 `SGLang / verl` 环境拆开

当前默认后端也固定为：

- baseline 多候选采样：`SGLang`
- RL rollout：`verl + SGLang`
- `vLLM` 只保留为后续对比路径

## 1. 开始前先确认三件事

### 1.1 GPU 正常

```bash
nvidia-smi
```

### 1.2 Python 3.12 可用

```bash
python3.12 --version
```

如果你的镜像没有 `python3.12`，先准备一个 Python 3.12 环境，再把 `PYTHON_BIN=python` 传给仓库脚本。

### 1.3 磁盘目录想清楚

建议直接采用：

- 持久化代码与模型：`/root/autodl-fs`
- 工作目录与输出：`/root/autodl-tmp/veridoc-rl`

## 2. 克隆与目录初始化

```bash
mkdir -p /root/autodl-fs/code
cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd VeriDoc-RL/VeriDoc-RL

mkdir -p /root/autodl-tmp/veridoc-rl/outputs
mkdir -p /root/autodl-tmp/veridoc-rl/pipelines
```

## 3. 环境变量

仓库提供了模板：

```bash
cp configs/autodl.env.example /tmp/veridoc_autodl.env
```

把里面这些值改成你自己的路径后：

```bash
source /tmp/veridoc_autodl.env
```

重点变量是：

- `VERIDOC_PROJECT_ROOT`
- `VERIDOC_WORK_ROOT`
- `VERIDOC_MODEL_PATH`
- `VERIDOC_TRAIN_PYTHON_BIN`
- `VERIDOC_RL_PYTHON_BIN`
- `VERIDOC_OUTPUT_ROOT`
- `VERIDOC_SFT_GOLD_PATH`
- `VERIDOC_RL_PROMPT_ONLY_PATH`
- `VERIDOC_API_BASE`

## 4. 重建双环境

默认入口：

```bash
bash scripts/bootstrap_autodl_envs.sh auto
```

如果你已经激活了 Python 3.12 环境：

```bash
PYTHON_BIN=python bash scripts/bootstrap_autodl_envs.sh auto
```

这个脚本会创建：

- `.venv-train`
  - `torch + .[dev,train]`
  - 用于 SFT / DPO / inference / pytest
- `.venv-rl`
  - `torch + sglang[srt] + pyarrow + verl`
  - 用于 `SGLang` serving 和 RL rollout

## 5. 先验环境检查

### 5.1 训练环境

```bash
source .venv-train/bin/activate
python -c "import torch, transformers, trl; print(torch.cuda.is_available(), torch.__version__, transformers.__version__, trl.__version__)"
pytest
veridoc-rl-smoke
```

### 5.2 RL / serving 环境

```bash
source .venv-rl/bin/activate
python -c "import torch, pyarrow, verl, sglang, fastapi, uvicorn, uvloop; print(torch.cuda.is_available(), torch.__version__, verl.__version__, sglang.__version__, uvicorn.__version__)"
```

## 6. 第一次跑通项目的推荐顺序

### 6.1 生成 SFT_gold

```bash
source .venv-train/bin/activate

python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path "${VERIDOC_SFT_GOLD_PATH}"
```

### 6.2 生成 RL_prompt_only

```bash
source .venv-train/bin/activate

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

脚本会优先读取 `VERIDOC_MODEL_PATH`。如果你想临时覆盖：

```bash
MODEL_PATH="${VERIDOC_MODEL_PATH}" bash scripts/start_sglang_server.sh
```

如果需要附加服务参数，可以直接往后追加：

```bash
bash scripts/start_sglang_server.sh --trust-remote-code
```

验证：

```bash
curl http://127.0.0.1:30000/v1/models
```

### 6.4 生成 baseline candidates

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

### 6.5 生成 preference 数据

```bash
source .venv-train/bin/activate

python scripts/generate_preference_dataset.py \
  --reference-path "${VERIDOC_SFT_GOLD_PATH}" \
  --candidate-path "${VERIDOC_WORK_ROOT}/outputs/candidates.jsonl" \
  --output-path "${VERIDOC_WORK_ROOT}/outputs/preferences.jsonl" \
  --min-margin 0.05
```

## 7. Pipeline 路线

仓库新增了一个 AutoDL 配置：

- [pipeline.autodl.qwen3_0p6.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_0p6.yaml)

这个配置会读取你 `source` 进去的环境变量。

### 7.1 先做 prepare-only

```bash
source .venv-train/bin/activate

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_0p6.yaml \
  --prepare-only
```

### 7.2 再正式执行

```bash
source .venv-train/bin/activate

python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_0p6.yaml
```

注意：

- 你仍然从 `.venv-train` 发起整条 pipeline
- `phase_c_*` 会自动切到 `VERIDOC_RL_PYTHON_BIN` 执行
- 这样双环境拆分才真正生效

## 8. 输出目录建议

建议你最终让这些内容都落在 `${VERIDOC_WORK_ROOT}` 下：

```text
${VERIDOC_WORK_ROOT}/
  outputs/
    sft_gold.jsonl
    rl_prompt_only.jsonl
    candidates.jsonl
    preferences.jsonl
  pipelines/
    qwen3_0p6_autodl/
```

## 9. 如果只想先做半程验证

最省时间的顺序是：

1. `bootstrap_autodl_envs.sh`
2. `pytest`
3. `generate_sft_dataset.py`
4. `start_sglang_server.sh`
5. `curl /v1/models`
6. `run_pipeline.py --prepare-only`

这条链路通了，再放开训练。

## 10. 旧的本地脚本

仓库里还保留：

- [rebuild_wsl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/rebuild_wsl_envs.sh)

但它现在属于旧的本地兼容脚本，不是默认入口。
