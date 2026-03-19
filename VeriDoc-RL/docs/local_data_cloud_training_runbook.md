# VeriDoc-RL 本地数据评测 + 云端训练执行清单

这份文档是独立 runbook，不替代全局 `README.md`。

目标是把流程拆成两部分：

- 本地机器：生成数据、构造 preference、准备语料、做评测、检查产物
- 云服务器：真正执行 `DPO` 和 `RL` 训练

适用前提：

- 你的本地显存大约 `10G`
- 你希望本地只做轻量流程，不在本地硬跑大模型训练
- 你接受当前仓库的 RL 仍然使用 `verifier reward`，不是独立 reward model

## 1. 先认清当前仓库能做什么

当前仓库已经提供：

- synthetic 数据生成
- Phase A 评测
- DPO preference 构造
- Phase A / B / C 训练语料准备
- Phase B DPO runtime bundle
- Phase C `verl` runtime bundle

当前仓库还**没有**提供：

- 候选生成器 / 推理脚本
- 模型训练完成后的统一推理脚本
- 独立 reward model 的训练 pipeline

这意味着：

- 你可以在本地完整生成 `reference`、`DPO_preference`、`phase_b_dpo`、`phase_c_rlvr`
- 但在构造 DPO preference 之前，你必须先准备一份外部生成的 `candidate predictions`
- 这份 `candidate predictions` 可以来自你自己的推理脚本、vLLM 服务、云端推理或手工构造的 smoke 数据

## 2. 路径约定

有两层根目录，不要混淆：

- git 根目录：`/home/alvis/projects/llm-study/VeriDoc-RL`
- Python 项目根目录：`/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL`

下面所有命令默认都在 **内层项目根目录** 执行：

```bash
cd /home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL
```

建议先建一个输出目录：

```bash
mkdir -p outputs
```

## 3. 本地环境准备

如果只是做数据、测试和评测：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

如果你还想本地做 DPO runtime 的 dry-run，但不真正训练，也建议装训练依赖：

```bash
pip install -e .[train]
```

如果只是本地 10G 机器，不建议在本地真正执行 RL 训练。

## 4. 本地先做基础检查

### 4.1 跑测试

```bash
pytest
```

### 4.2 跑最小 smoke

```bash
veridoc-rl-smoke
```

如果这两步失败，不要继续往后走。

## 5. 本地生成 reference 数据

### 5.1 生成 SFT / reference 数据

这一步产物会同时包含：

- `input`
- `reference`
- `metadata`

命令：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path outputs/sft_gold.jsonl
```

参数说明：

- `--count`：生成多少条样本
- `--seed`：随机种子，固定后可复现
- `--task-type`：只能选 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
- `--output-path`：输出 JSONL 路径

如果你想限制 bucket，可以重复传这些参数：

- `--template-family template_a`
- `--template-family template_b`
- `--ocr-noise-level low`
- `--ocr-noise-level medium`
- `--hard-case-type field_missing`
- `--rule-complexity cross_field`

这些参数都支持多次传入。脚本会在你给定的候选值上循环采样。

示例：只生成 `template_a/template_b`，并限制在 `low/medium` OCR 噪声：

```bash
python scripts/generate_sft_dataset.py \
  --count 60 \
  --seed 7 \
  --task-type SFT_gold \
  --template-family template_a \
  --template-family template_b \
  --ocr-noise-level low \
  --ocr-noise-level medium \
  --output-path outputs/sft_gold.small.jsonl
```

### 5.2 生成 RL prompt-only 数据

RL 阶段使用的上游数据通常是 `RL_prompt_only`：

```bash
python scripts/generate_sft_dataset.py \
  --count 200 \
  --seed 11 \
  --task-type RL_prompt_only \
  --output-path outputs/rl_prompt_only.jsonl
```

注意：

- `RL_prompt_only` 默认不包含 `reference`
- 如果你只是想让训练框架跑起来，这没有问题
- 如果你希望后续做离线检查，建议额外保留一份 `SFT_gold`

## 6. 本地做 Phase A 评测

这一步不是训练，只是验证你的 prediction 文件和 reference 是否对齐。

### 6.1 准备 prediction 文件

仓库没有推理脚本，所以 `outputs/predictions.jsonl` 需要你自己准备。

最小格式建议如下，每行一个对象：

```json
{
  "sample_id": "template_a_00000",
  "fields": {
    "policyholder_name": "张三"
  },
  "validations": []
}
```

如果你愿意，也可以显式包成：

```json
{
  "prediction": {
    "sample_id": "template_a_00000",
    "fields": {
      "policyholder_name": "张三"
    },
    "validations": []
  }
}
```

### 6.2 运行评测

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/sft_gold.jsonl \
  --prediction-path outputs/predictions.jsonl \
  --report-path outputs/phase_a_report.json \
  --case-export-path outputs/phase_a_cases.jsonl \
  --failure-only
```

参数说明：

- `--reference-path`：必须是带 `reference` 的 JSONL，通常就是 `SFT_gold`
- `--prediction-path`：你的模型输出
- `--report-path`：汇总报告 JSON
- `--case-export-path`：逐 case 导出
- `--failure-only`：只导出失败样本
- `--failure-case-limit`：默认 `3`，控制 summary 里保留多少失败样本
- `--reward-profile`：默认 `default`，可用于做 reward ablation

### 6.3 对比多份评测报告

```bash
python scripts/compare_phase_reports.py \
  --report baseline=outputs/phase_a_report.json \
  --report another=outputs/phase_a_report_2.json \
  --output-dir outputs/report_compare
```

参数说明：

- `--report`：格式必须是 `label=path`
- `--output-dir`：输出对比 JSON、Markdown 和 SVG 图表
- `--bucket-dimension`：默认 `ocr_noise_level`
- `--bucket-metric`：默认 `field_f1`

## 7. 本地准备 DPO 的 candidate predictions

这是最容易卡住的一步。

当前仓库**不提供 candidate sampler**，所以你必须自己准备 `candidate.jsonl`。

### 7.1 candidate 文件最小要求

每一行代表某个样本的一条候选输出，最小格式：

```json
{
  "candidate_id": "qwen25-3b-sample0-cand0",
  "prediction": {
    "sample_id": "template_a_00000",
    "fields": {
      "policyholder_name": "张三"
    },
    "validations": []
  }
}
```

关键要求：

- 至少要有 `candidate_id`
- 必须有 `prediction.sample_id`
- 同一个 `sample_id` 至少要有 **2 条候选**
- `sample_id` 必须能和 `reference-path` 里的样本对齐

你也可以附带：

- `metadata`
- `context`

但不是必需的。

### 7.2 一份可用的 candidate 示例

假设某个样本 `template_a_00000`，你至少要有类似下面两行：

```json
{"candidate_id":"good-0","prediction":{"sample_id":"template_a_00000","fields":{"policyholder_name":"张三","policyholder_phone":"13800138000"},"validations":[]}}
{"candidate_id":"bad-0","prediction":{"sample_id":"template_a_00000","fields":{"policyholder_name":"张三","policyholder_phone":""},"validations":[]}}
```

### 7.3 候选如何生成

你可以用以下任一方式：

- 你自己的推理脚本，读取 `outputs/sft_gold.jsonl` 里的 `input`，调用模型生成 JSON
- 云端 vLLM / Transformers 推理后，把结果写回本地 JSONL
- 手工构造少量 smoke 数据，用来验证 preference pipeline

如果你只是想先验证整个 DPO 链路，推荐先做一个很小的 smoke 集：

- 20 条 reference
- 每条 2 个 candidate
- 一个明显更好，一个明显更差

## 8. 本地生成 DPO preference 数据

有了 `reference + candidate` 之后，运行：

```bash
python scripts/generate_preference_dataset.py \
  --reference-path outputs/sft_gold.jsonl \
  --candidate-path outputs/candidate.jsonl \
  --output-path outputs/preferences.jsonl \
  --reward-profile default \
  --min-margin 0.05 \
  --max-pairs-per-sample 1
```

参数说明：

- `--reference-path`：通常用 `SFT_gold`
- `--candidate-path`：你自己准备的候选输出
- `--output-path`：导出的 `DPO_preference`
- `--reward-profile`：用于排序候选的 verifier reward profile
- `--min-margin`：chosen 和 rejected 的最小 reward 差值，太小的 pair 会被丢弃
- `--max-pairs-per-sample`：每个样本最多导出多少组 preference
- `--include-all-pairs`：如果加上这个开关，会导出更多 pair，而不是只取最佳对最差

推荐默认值：

- `--min-margin 0.05`
- `--max-pairs-per-sample 1`

如果你的 candidate 很接近，pair 数量可能很少，这通常说明：

- 候选质量差异不够大
- `min-margin` 设得太高

## 9. 本地准备训练语料

### 9.1 准备 Phase B DPO 语料

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/preferences.jsonl \
  --output-path outputs/train.phase_b_dpo.jsonl \
  --stage phase_b_dpo
```

这一步会把 `DPO_preference` 转成训练语料，包含：

- `system_prompt`
- `prompt`
- `chosen`
- `rejected`

### 9.2 准备 Phase C RLVR 语料

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/rl_prompt_only.jsonl \
  --output-path outputs/train.phase_c_rlvr.jsonl \
  --stage phase_c_rlvr \
  --reward-profile rlvr
```

参数说明：

- `--stage`：只能选 `phase_a_sft`、`phase_b_dpo`、`phase_c_rlvr`
- `--system-prompt`：如不传，使用仓库默认 system prompt
- `--reward-profile`：只对 RL corpus 有意义，默认 `rlvr`

## 10. 本地生成训练 manifest

### 10.1 为 DPO 生成 manifest

```bash
python scripts/generate_training_manifests.py \
  --matrix-path configs/experiment_matrix.yaml \
  --train-data-path outputs/train.phase_b_dpo.jsonl \
  --output-dir outputs/training_bundle_dpo \
  --base-model Qwen2.5-3B-Instruct
```

### 10.2 为 RL 生成 manifest

```bash
python scripts/generate_training_manifests.py \
  --matrix-path configs/experiment_matrix.yaml \
  --train-data-path outputs/train.phase_c_rlvr.jsonl \
  --output-dir outputs/training_bundle_rl \
  --base-model Qwen2.5-3B-Instruct
```

参数说明：

- `--matrix-path`：实验矩阵
- `--train-data-path`：上一步生成的训练语料
- `--output-dir`：manifest bundle 目录
- `--eval-data-path`：可选评测语料
- `--base-model`：强烈建议显式传，不要完全依赖 matrix 默认值

如果你本地只是 smoke，建议 `Qwen2.5-1.5B-Instruct`。  
如果你上云做正式 DPO，可从 `Qwen2.5-3B-Instruct` 起步。  
如果你要做 RL，建议把更大的模型留给云端。

## 11. 本地检查 runtime bundle，但不真正训练

### 11.1 DPO runtime dry-run

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_dpo/phase_b_dpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_b_dpo
```

你应该看到这些产物：

- `outputs/runtime_runs/phase_b_dpo/runtime_plan.json`
- `outputs/runtime_runs/phase_b_dpo/launch.sh`
- `outputs/runtime_runs/phase_b_dpo/dpo_config.json`
- `outputs/runtime_runs/phase_b_dpo/data/phase_b_dpo.train.jsonl`

### 11.2 RL runtime dry-run

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_c_grpo \
  --materialize-data
```

这一步如果失败，最常见原因是本地没有 `pyarrow`。  
如果你只想本地检查，不想装它，也可以先不加 `--materialize-data`。

## 12. 需要上传到云端的内容

最稳妥的做法是把**整个仓库**推到云端，而不是只传几个 JSON。

原因：

- DPO runtime 通过 `python -m veridoc_rl.training.trl_dpo` 执行
- RL runtime 通过 `custom_reward_function.path=.../verl_reward.py` 读取仓库内 reward bridge
- 只传 manifest 不够，云端还需要同一份源码

最小需要带上的内容：

- 整个仓库源码
- `outputs/train.phase_b_dpo.jsonl`
- `outputs/train.phase_c_rlvr.jsonl`
- `outputs/training_bundle_dpo/`
- `outputs/training_bundle_rl/`

如果你不想上传整个 `outputs/`，至少带这几项。

## 13. 云端执行 DPO

### 13.1 建议的云端机器

如果你只做 DPO：

- 最低建议：`24G` 单卡
- 更稳妥：`48G` 单卡

### 13.2 云端环境

进入项目根目录后：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train]
```

注意：

- `.[train]` 不包含 `torch`
- 你需要按云服务器 CUDA 版本单独安装 `torch`

### 13.3 生成 DPO runtime bundle

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_dpo/phase_b_dpo/manifest.json \
  --run-dir outputs/cloud_runs/phase_b_dpo
```

### 13.4 真正执行 DPO

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_dpo/phase_b_dpo/manifest.json \
  --run-dir outputs/cloud_runs/phase_b_dpo \
  --execute
```

建议第一次先不执行，先检查：

- `runtime_plan.json`
- `dpo_config.json`
- `launch.sh`

### 13.5 DPO 常见调参入口

如果你要调 DPO 训练参数，优先改：

- `configs/experiment_matrix.yaml`
- 或在生成 manifest 时指定更合适的 `--base-model`

当前 DPO 默认值包括：

- `beta=0.1`
- `epochs=1`
- `learning_rate=1e-6`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`

## 14. 云端执行 RL

### 14.1 建议的云端机器

如果你要跑 `GRPO` / `RLOO`：

- 不建议用本地 10G
- 单卡建议至少 `48G`
- 更重的实验建议更高档位云卡，或多卡

### 14.2 云端环境

除了基础依赖外，还要有：

- `verl`
- `pyarrow`
- 通常还需要 `vllm`
- 与 CUDA 版本匹配的 `torch`

### 14.3 生成 RL runtime bundle

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/cloud_runs/phase_c_grpo \
  --materialize-data
```

### 14.4 真正执行 RL

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/cloud_runs/phase_c_grpo \
  --materialize-data \
  --execute
```

如果你想跑 `RLOO`，把 manifest 路径替换成：

```text
outputs/training_bundle_rl/phase_c_rloo/manifest.json
```

## 15. 训练完之后如何评测

当前仓库没有“训练完自动推理并回写评测”的统一脚本，所以你需要自己补最后两步：

1. 用训练后的 checkpoint 做推理，生成 `prediction.jsonl`
2. 重新调用评测脚本

评测命令仍然是：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/sft_gold.jsonl \
  --prediction-path outputs/predictions.from_trained_model.jsonl \
  --report-path outputs/post_train_report.json \
  --case-export-path outputs/post_train_cases.jsonl \
  --failure-only
```

如果要对比多次实验：

```bash
python scripts/compare_phase_reports.py \
  --report before=outputs/phase_a_report.json \
  --report after=outputs/post_train_report.json \
  --output-dir outputs/post_train_compare
```

## 16. verifier reward 和 verl 是否冲突

结论：**不冲突，当前实现是兼容的。**

### 16.1 为什么不冲突

`verl` 官方支持两种路线：

- 使用 reward model
- 使用自定义 reward function

对自定义 reward function，官方要求的函数参数是：

- `data_source`
- `solution_str`
- `ground_truth`
- `extra_info`

本仓库的 [src/veridoc_rl/training/verl_reward.py](../src/veridoc_rl/training/verl_reward.py) 里的 `compute_score()` 正好就是这个接口。

另外，本仓库在 RL runtime 中显式传入：

- `reward_model.reward_manager=naive`
- `custom_reward_function.path=.../verl_reward.py`
- `custom_reward_function.name=compute_score`

这意味着当前 RL 走的是：

- `verl` 负责 rollout / PPO-style RL
- 本仓库自己的 verifier 负责把模型输出转成 reward 分数

所以这不是“拿 verifier reward 强行替代 RM 导致框架冲突”，而是使用了 `verl` 官方支持的 **custom reward function** 路线。

### 16.2 当前方案的真实含义

当前 RL 不是经典意义上的“learned reward model RLHF”。

它更准确地说是：

- rule-based / verifier-based reward optimization
- 用程序化规则直接给 rollout 打分

优点：

- 可解释
- 可控
- 不需要先训练 RM

缺点：

- reward 设计上限受 verifier 质量限制
- reward 计算可能成为 Python 侧瓶颈
- 无法学习 verifier 没覆盖到的细粒度偏好

### 16.3 当前实现的注意事项

虽然不冲突，但你需要注意：

- 云端机器上必须能访问到仓库里的 `verl_reward.py`
- 如果是多机训练，所有节点最好使用同一份代码路径或共享文件系统
- `compute_score()` 依赖模型输出能被解析成 JSON；格式很差时不会报错退出，而是回退到空字段并给低 reward
- 这条路线并不等价于“已经支持 reward model 训练”

## 17. 建议的实际执行顺序

如果你现在马上要开始做实验，建议按这个顺序：

1. 本地 `pytest`
2. 本地生成 `SFT_gold`
3. 本地准备少量 `candidate.jsonl`
4. 本地生成 `preferences.jsonl`
5. 本地生成 `train.phase_b_dpo.jsonl`
6. 本地生成 `train.phase_c_rlvr.jsonl`
7. 本地生成 `training_bundle_dpo` 和 `training_bundle_rl`
8. 本地 dry-run `prepare_training_runtime.py`
9. 把整个仓库和 `outputs/` 关键产物上传到云端
10. 云端先跑 DPO
11. 云端再跑 RL
12. 训练后自行推理，回到本地做 Phase A 评测和报告对比

## 18. 对你这台 10G 本地机的直接建议

你这台机器最适合：

- 跑所有数据脚本
- 跑评测
- 跑对比分析
- 检查 manifest / runtime bundle
- 做很小规模 smoke

不适合：

- 正式 DPO 训练
- 正式 RL 训练

如果只是做本地 smoke，模型建议控制在：

- `Qwen2.5-1.5B-Instruct`

更正式的 DPO / RL，请直接放云端。
