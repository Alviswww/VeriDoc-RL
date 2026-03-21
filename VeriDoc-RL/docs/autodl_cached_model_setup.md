# AutoDL 方案 B：先缓存模型快照，再走本地模型目录

这是更稳的方案。核心思路是先把 `Qwen/Qwen3-1.7B` 固化到本地目录，再让 `SGLang` 和 `transformers` 都读同一份模型。

适用场景：

- 你准备长期在同一台 AutoDL 机器上训练
- 你不希望每次重启都重新走在线缓存
- 你想把模型和实验输出彻底分开

## 1. 先完成基础环境

先执行到双环境安装完成：

```bash
bash scripts/bootstrap_autodl_envs.sh auto all
source .venv-train/bin/activate
```

如果你还没配置环境变量，先复制模板：

```bash
cp configs/autodl.env.example /tmp/veridoc_autodl.env
source /tmp/veridoc_autodl.env
```

## 2. 下载模型快照

建议把模型放在持久化磁盘：

```bash
mkdir -p /root/autodl-fs/models

python scripts/prefetch_hf_model.py \
  --model-id Qwen/Qwen3-1.7B \
  --target-dir /root/autodl-fs/models/Qwen3-1.7B
```

完成后你会得到一个完整目录：

```text
/root/autodl-fs/models/Qwen3-1.7B
```

## 3. 切换环境变量到本地模型

把环境文件里的模型变量改成：

```bash
export VERIDOC_MODEL_REF="/root/autodl-fs/models/Qwen3-1.7B"
export VERIDOC_MODEL_PATH="${VERIDOC_MODEL_REF}"
```

重新加载：

```bash
source /tmp/veridoc_autodl.env
```

## 4. 启动 SGLang

```bash
bash scripts/start_sglang_server.sh
```

验证：

```bash
curl http://127.0.0.1:30000/v1/models
```

## 5. 生成 baseline candidates

```bash
source .venv-train/bin/activate

python scripts/generate_candidates.py \
  --input-path "${VERIDOC_SFT_GOLD_PATH}" \
  --output-path "${VERIDOC_WORK_ROOT}/outputs/candidates.jsonl" \
  --model "${VERIDOC_MODEL_REF}" \
  --api-base "${VERIDOC_API_BASE}" \
  --num-candidates 4 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 1024
```

## 6. 跑完整 pipeline

先 prepare-only：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

再正式执行：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml
```

## 7. 这个方案的优势

- `SGLang` 和 `transformers` 读取的是同一份模型目录
- 模型不依赖临时缓存目录
- 更适合做多次重启、多轮训练、长期保留 checkpoint

## 8. 推荐目录布局

```text
/root/autodl-fs/
  code/
    VeriDoc-RL/
  models/
    Qwen3-1.7B/
  .cache/
    huggingface/

/root/autodl-tmp/
  veridoc-rl/
    outputs/
    pipelines/
```

## 9. 什么时候选这个方案

- 你已经确认方案 A 可以跑
- 你希望环境更稳、更可复现
- 你准备开始真正的训练与多轮实验
