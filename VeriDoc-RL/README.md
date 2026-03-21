# VeriDoc-RL

VeriDoc-RL 是一个面向制式投保单结构化抽取的 verifier-guided post-training 项目。当前仓库已经完全收敛到云端单机 GPU 主线，默认服务与训练路线如下：

- baseline：`Qwen/Qwen3-1.7B`
- baseline candidate generation：`SGLang`
- `phase_a_sft`：`transformers + peft`
- `phase_b_dpo`：`TRL DPOTrainer`
- `phase_c_*`：`verl + SGLang`
- 默认环境：`VERIDOC_WORK_ROOT` 下的 `.venv-train` 和 `.venv-rl` 双环境拆分

## 当前默认配置

- 主配置：[pipeline.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.qwen3_1p7.yaml)
- AutoDL 配置：[pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml)
- 实验矩阵：[experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml)
- 环境模板：[autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example)

当前训练默认也同步切到了更适合 4090 / 5090 的 `bfloat16 + QLoRA` 组合。

## 执行文档

推荐先看 runbook，再决定是走在线模型还是本地快照：

- 详细执行手册：[autodl_runbook.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_runbook.md)
  - 面向第一次接触仓库的使用者
  - 覆盖无卡安装、开卡验证、SGLang 启动、数据生成、prepare-only、完整 pipeline
- 方案 A，最快起步：[autodl_online_hf_setup.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_online_hf_setup.md)
  - 直接使用 `Qwen/Qwen3-1.7B`
  - 优点是省磁盘、准备快、最适合第一次在 AutoDL 验证全链路
- 方案 B，更稳更可复现：[autodl_cached_model_setup.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_cached_model_setup.md)
  - 先把模型快照下载到 `/root/autodl-fs/models/Qwen3-1.7B`
  - 优点是重启实例后更稳定，也更适合长跑训练

## 快速开始

```bash
mkdir -p /root/autodl-fs/code
cd /root/autodl-fs/code
git clone <your-repo-url> VeriDoc-RL
cd VeriDoc-RL/VeriDoc-RL

cp configs/autodl.env.example /tmp/veridoc_autodl.env
# 按你的路径修改 /tmp/veridoc_autodl.env 后再 source
source /tmp/veridoc_autodl.env

bash scripts/bootstrap_autodl_envs.sh auto all
source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
pytest
veridoc-rl-smoke
```

然后启动服务并先跑 prepare-only：

```bash
bash scripts/start_sglang_server.sh

source "${VERIDOC_WORK_ROOT}/.venv-train/bin/activate"
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

## 关键脚本

- [bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh)
  - 默认重建 `VERIDOC_WORK_ROOT` 下的 `.venv-train` 和 `.venv-rl`
  - 支持 `all | train | rl`
- [start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh)
  - 支持 Hugging Face repo id 或本地模型目录
  - 优先读取 `MODEL_REF / VERIDOC_MODEL_REF`
  - 同时兼容 `VERIDOC_RL_PYTHON_BIN`
- [prefetch_hf_model.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/prefetch_hf_model.py)
  - 把 HF 模型快照固化到本地目录

## 目录约定

建议统一采用：

- 持久化代码、模型、HF cache：`/root/autodl-fs`
- 工作目录、输出、临时 cache：`/root/autodl-tmp`

在 AutoDL 上，最稳定的分工是：

- `.venv-train`
  - pytest
  - synthetic data
  - SFT / DPO
  - checkpoint inference
- `.venv-rl`
  - SGLang serving
  - `verl` rollout

## 依赖边界

`Qwen3` 需要 `transformers>=4.51`，仓库已经同步上调依赖下限。`SGLang` 与 `verl` 仍然建议放在单独环境中，不再尝试维护本地 WSL 的兼容路线。
