# VeriDoc-RL 本地单机执行清单

这个文件保留了原来的路径名，但内容已经改成 **本地单机模式 runbook**。当前仓库的默认路线不再是假设“本地准备数据、云端训练”，而是：

- 在同一台 Windows + WSL 机器里完成环境安装
- 在 WSL 里启动 `SGLang`
- 在 WSL 里执行 `SFT / DPO / RL`
- 在 WSL 里回评和对比

对 `RTX 2060 6GB / SM75` 这类机器，当前默认版本岛已经收敛到：

- `torch 2.6.0`
- `sglang 0.4.6.post5`
- `verl 0.4.1`
- 默认训练精度 `float16`

如果你只想照着一份执行清单把项目跑起来，这份文档比全局 `README.md` 更偏操作。

## 1. 先认清当前项目的框架分工

先把最关键的边界说清楚：

- baseline candidate 生成：`SGLang`
- checkpoint 回评推理：`transformers + peft`
- `phase_a_sft`：`transformers.Trainer + datasets + peft`
- `phase_b_dpo`：`TRL DPOTrainer`
- `phase_c_grpo / phase_c_rloo`：`verl`

所以：

- `SGLang` 默认只负责 baseline candidate generation
- SFT 训练框架不是 `TRL SFTTrainer`
- 当前 SFT 实现是仓库里的 `src/veridoc_rl/training/trl_sft.py`
- 这个文件名有历史原因，但内部实际用的是 `transformers.Trainer`

## 2. 推荐主线

当前本地单机主线固定为：

1. baseline：`models/Qwen3-0.6B`
2. `phase_a_sft`：在 baseline 上做 `QLoRA SFT`
3. `phase_b_dpo`：在 `phase_a_sft/checkpoints` 上做 `DPO`
4. `phase_c_grpo` 或 `phase_c_rloo`：在 `phase_a_sft/checkpoints` 上做 `RLVR`
5. 对比 `baseline / sft / dpo / rlvr`

## 3. 目录约定

你需要记住两层根目录：

- git 根目录：`/home/alvis/projects/llm-study/VeriDoc-RL`
- Python 项目根目录：`/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL`

下面所有命令默认都在内层项目根目录执行：

```bash
cd /home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL
```

## 4. WSL 环境准备

### 4.1 原则

统一在 WSL 里做下面这些事：

- Python 环境
- 依赖安装
- `SGLang` 启动
- 训练
- 推理
- 评测

不要把脚本拆成一半在 PowerShell、一半在 WSL 跑。

### 4.2 Python 版本

当前仓库要求 `Python >= 3.12`。  
推荐直接使用 `Python 3.12`。

如果 WSL 里已经有 `python3.12`：

```bash
python3.12 -m venv .venv-rl
```

如果你准备用 `uv`：

```bash
uv python install 3.12
uv venv --python 3.12 .venv-rl
```

### 4.3 GPU 检查

正式开始前，先确认 WSL 能拿到 Windows 的 GPU：

```bash
nvidia-smi
```

这一步必须正常返回显卡信息。如果返回类似：

```text
Failed to initialize NVML: GPU access blocked by the operating system
```

说明你当前 WSL 会话还没有拿到 GPU，不要继续往下装训练栈。

### 4.4 安装依赖

当前默认主路径只需要一个环境：

- `.venv-rl`
  - 跑仓库代码、`prepare-only`、SFT、DPO、RL
  - baseline candidate generation 默认也通过这个环境对接本地 `SGLang`
  - 当前默认 RL rollout backend 是 `sglang`

如果你后续想单独做 `vLLM` 对比，再额外准备 `.venv-vllm`。

对于 `RTX 2060 6GB` 这类老卡，当前仓库默认训练精度已经收敛到 `float16`，不要再把默认精度改回 `bfloat16`。  
当前默认也不再追最新的 `SGLang / verl / torch`，而是优先使用对 `SM75` 更稳的版本岛。

先确认一件事：`/usr/local/cuda` 必须先切到 `12.6.x` 或 `12.4.x`，再重建环境。

当前这台仓库机器上观测到的是：

```text
/usr/local/cuda -> /usr/local/cuda-13.2
```

如果你的机器也是 13.x，先不要继续复用旧环境，先把用户态 CUDA 收敛到 `12.6.x` 或 `12.4.x`。

仓库里已经提供了重建脚本：

```bash
bash scripts/rebuild_wsl_envs.sh cu126
```

说明：

- 也可以把 `cu126` 换成 `cu124`
- 脚本会先检查 `/usr/local/cuda/version.json` 是否匹配目标大版本
- `.venv-rl` 会安装 `torch 2.6.0 + verl 0.4.1 + sglang 0.4.6.post5`
- 默认不会碰 `.venv-vllm`
- `SGLang` 会通过 FlashInfer 的 `torch2.6` wheel 源安装依赖
- 对 `SM75` 卡，默认建议用 `--attention-backend triton --sampling-backend pytorch` 启服务
- 如果你必须保留自己的 `torch + cu126/cu124` 组合，不要直接装 `vLLM` 预编译 wheel；它会带自己的 `torch` 依赖
- 只有启用 `--with-vllm` 时，脚本才要求本机可用 `uv`
- 如果 vLLM 编译过程报 `Killed` 或退出码 `137`，优先增大 WSL 的 `memory` 和 `swap`
- `vLLM` 和 `verl` 不建议强行塞进同一个环境
- `cu118` 先不作为默认路线；只有当前 `cu126/cu124 + torch2.6 + sglang0.4.6.post5` 仍然失败时，再把它当最后一级 fallback
- 如果你只想先做 dry-run，也建议直接用 `.venv-rl`

### 4.5 快速验证依赖

```bash
source .venv-rl/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers, datasets, peft, trl; print('train-ok')"
python -c "import pyarrow, verl, sglang; print('rl-ok')"
```

## 5. 输出目录建议

建议你一开始就约定好输出目录：

```text
outputs/
  sft_gold.jsonl
  rl_prompt_only.jsonl
  candidates.jsonl
  preferences.jsonl
  train.phase_a_sft.jsonl
  train.phase_b_dpo.jsonl
  train.phase_c_rlvr.jsonl
  training_bundle/
  runtime_runs/
  pipelines/
```

先建目录：

```bash
mkdir -p outputs
```

## 6. 第一次启动项目的推荐顺序

### 6.1 先跑测试

```bash
pytest
veridoc-rl-smoke
```

### 6.2 生成数据

生成 `SFT_gold`：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path outputs/sft_gold.jsonl
```

生成 `RL_prompt_only`：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 11 \
  --task-type RL_prompt_only \
  --output-path outputs/rl_prompt_only.jsonl
```

### 6.3 启动 SGLang

开一个终端，保持服务常驻：

```bash
source .venv-rl/bin/activate

python -m sglang.launch_server \
  --model-path models/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000 \
  --attention-backend triton \
  --sampling-backend pytorch
```

再开一个终端做检查：

```bash
source .venv-rl/bin/activate
curl http://127.0.0.1:30000/v1/models
```

### 6.4 手工生成 candidates

```bash
python scripts/generate_candidates.py \
  --input-path outputs/sft_gold.jsonl \
  --output-path outputs/candidates.jsonl \
  --model models/Qwen3-0.6B \
  --api-base http://127.0.0.1:30000/v1 \
  --num-candidates 4 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 1024
```

### 6.5 生成 preferences

```bash
python scripts/generate_preference_dataset.py \
  --reference-path outputs/sft_gold.jsonl \
  --candidate-path outputs/candidates.jsonl \
  --output-path outputs/preferences.jsonl \
  --min-margin 0.05
```

### 6.6 导出训练语料

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/sft_gold.jsonl \
  --output-path outputs/train.phase_a_sft.jsonl \
  --stage phase_a_sft

python scripts/prepare_training_data.py \
  --input-path outputs/preferences.jsonl \
  --output-path outputs/train.phase_b_dpo.jsonl \
  --stage phase_b_dpo

python scripts/prepare_training_data.py \
  --input-path outputs/rl_prompt_only.jsonl \
  --output-path outputs/train.phase_c_rlvr.jsonl \
  --stage phase_c_rlvr \
  --reward-profile rlvr
```

### 6.7 生成 manifest

```bash
python scripts/generate_training_manifests.py \
  --matrix-path configs/experiment_matrix.yaml \
  --phase-a-train-data-path outputs/train.phase_a_sft.jsonl \
  --phase-b-train-data-path outputs/train.phase_b_dpo.jsonl \
  --phase-c-train-data-path outputs/train.phase_c_rlvr.jsonl \
  --output-dir outputs/training_bundle \
  --phase-a-base-model models/Qwen3-0.6B \
  --phase-b-base-model outputs/runtime_runs/phase_a_sft/checkpoints \
  --phase-c-base-model outputs/runtime_runs/phase_a_sft/checkpoints
```

### 6.8 生成 runtime bundle

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle/phase_a_sft/manifest.json \
  --run-dir outputs/runtime_runs/phase_a_sft

python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle/phase_b_dpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_b_dpo

python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle/phase_c_grpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_c_grpo \
  --materialize-data
```

### 6.9 先做一次 prepare-only 总编排

如果你第一次跑，建议先不要直接上真实训练，先看整个主线能不能产出正确的 bundle：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.qwen3_0p6.yaml \
  --prepare-only
```

## 7. 一键本地主线

如果你已经准备好了：

- `outputs/sft_gold.jsonl`
- `outputs/rl_prompt_only.jsonl`
- 本地 `SGLang` 服务
- `.venv-rl`

可以直接执行：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.qwen3_0p6.yaml
```

它会自动完成：

- baseline candidate generation
- baseline eval
- `phase_a_sft`
- `phase_b_dpo`
- `phase_c_grpo` 或 `phase_c_rloo`
- `state.json`
- `summary.json`

## 8. pipeline 产物应该怎么看

默认运行目录：

```text
outputs/pipelines/qwen3_0p6_mainline/
```

关键文件：

- `spec.snapshot.json`
- `state.json`
- `summary.json`

### 8.1 `state.json`

这个文件记录：

- 每个阶段的状态
- 当前使用的 base model
- 训练语料路径
- manifest 路径
- runtime plan 路径
- checkpoint 路径
- prediction / report / cases 路径

### 8.2 `summary.json`

这个文件记录：

- 当前 pipeline 总状态
- 各阶段摘要
- 如果有 report，则会写入 comparison 结果

### 8.3 阶段目录

一个典型阶段目录会长这样：

```text
phase_a_sft/
  train.jsonl
  manifest.json
  runtime_plan.json
  launch.sh
  checkpoints/
  predictions.jsonl
  report.json
  cases.jsonl
```

## 9. checkpoint 依赖关系

当前本地编排层已经固定好这组规则：

- baseline 用 `models/Qwen3-0.6B`
- `phase_a_sft` 用 `models/Qwen3-0.6B`
- `phase_b_dpo` 用 `phase_a_sft/checkpoints`
- `phase_c_*` 用 `phase_a_sft/checkpoints`

这就是当前推荐主线，不要手工把 `phase_c` 改成默认接在 `phase_b_dpo` 后面。

## 10. 训练后回评

如果你不走一键编排，而是自己手工做训练，可以用统一推理入口把 checkpoint 导回评测链路：

```bash
python scripts/run_inference.py \
  --input-path outputs/sft_gold.jsonl \
  --output-path outputs/predictions.sft.jsonl \
  --model-name-or-path outputs/runtime_runs/phase_a_sft/checkpoints
```

再跑评测：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/sft_gold.jsonl \
  --prediction-path outputs/predictions.sft.jsonl \
  --report-path outputs/report.sft.json \
  --case-export-path outputs/cases.sft.jsonl
```

对比四阶段结果：

```bash
python scripts/compare_phase_reports.py \
  --report baseline=outputs/report.baseline.json \
  --report sft=outputs/report.sft.json \
  --report dpo=outputs/report.dpo.json \
  --report rlvr=outputs/report.rlvr.json \
  --output-dir outputs/report_compare
```

## 11. 低资源机器怎么调

### 11.1 baseline / candidate generation

- 保持 `num_candidates=4`
- 优先降低 `max_new_tokens`
- 不要先把多候选降成单候选

### 11.2 SFT / DPO

- 保持 `QLoRA`
- 降 `per_device_train_batch_size`
- 降 `max_length`
- 升 `gradient_accumulation_steps`

### 11.3 RL

- 先降 `rollout_n`
- 再降 `max_response_length`
- 再降 `ppo_micro_batch_size_per_gpu`

## 12. 常见问题

### Q1. 为什么 README 里会同时出现 SGLang、transformers 和 verl

因为它们解决的是不同问题：

- `SGLang` 负责默认 baseline 多候选采样
- `transformers` 负责训练和 checkpoint 本地推理
- `verl + sglang` 负责 RL rollout

### Q2. 为什么现在默认不是更高版本的 SGLang / verl / torch

因为当前目标不是追最新版，而是先让 `RTX 2060 / SM75` 这种老卡本地跑通。  
所以主路径先固定在：

- `torch 2.6.0`
- `sglang 0.4.6.post5`
- `verl 0.4.1`
- `float16`

### Q3. `trl_sft.py` 为什么不是 TRL 的 SFTTrainer

这个文件名保留了历史命名，但当前实现里用的是：

- `datasets.Dataset`
- `transformers.Trainer`
- `peft`

### Q4. 现在有没有 reward model

没有。  
当前 RL 仍默认使用 verifier-based reward。

### Q5. 第一次启动时要不要先把模型下载到本地

不是必须。  
当前仓库默认已经直接写成本地路径 `models/Qwen3-0.6B`；只有你手工改回 `Qwen/Qwen3-0.6B` 时，才会发生按需下载。

## 13. 推荐的第一次实战节奏

如果你现在准备在 Windows + WSL 上真正开跑，建议节奏是：

1. 先解决 `nvidia-smi` 在 WSL 内可用
2. 先把 `/usr/local/cuda` 切到 `12.6.x` 或 `12.4.x`
3. 运行 `bash scripts/rebuild_wsl_envs.sh cu126`
4. 跑 `pytest`
5. 跑 `prepare-only pipeline`
6. 再跑完整 pipeline

如果你只想先把项目跑通，不建议第一天就直接上 RL。更稳的顺序是：

1. baseline
2. `phase_a_sft`
3. `phase_b_dpo`
4. 最后再补 `phase_c_grpo`
