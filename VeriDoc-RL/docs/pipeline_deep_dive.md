# VeriDoc-RL Pipeline 深度解读

这份文档不是操作手册，而是 **完整理解仓库 pipeline 的说明书**。

目标是回答下面这些问题：

- 这个项目到底在做什么。
- 数据一开始长什么样，后面如何一步步变成训练数据。
- `SFT / DPO / RL` 三个阶段分别在学什么。
- `prepare-only` 到底做了什么，为什么也会占 GPU。
- 每个阶段对应哪些文件，输入输出分别是什么。
- 大概会消耗多少资源。

如果你第一次接触仓库，建议先配合下面这些文件一起看：

- [autodl_runbook.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/docs/autodl_runbook.md)
- [scripts/README.md](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/README.md)
- [configs/pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml)
- [configs/experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml)

## 1. 这个项目到底在做什么

一句话概括：

> 给模型一页投保单的 OCR token，让它输出结构化字段 `fields`，同时输出规则校验结果 `validations`，再用 verifier reward 持续优化它。

这个项目的目标不是普通的“只抽字段”。

它希望模型一次性完成两件事：

1. 抽取投保单字段。
2. 对抽取结果做规则校验。

所以模型的目标输出格式始终是一个 JSON：

```json
{
  "sample_id": "template_a_00000",
  "fields": {
    "policyholder_name": "王敏",
    "insured_birth_date": "1990-01-01"
  },
  "validations": [
    {
      "rule_id": "required.policyholder_name",
      "status": "pass",
      "message": "policyholder_name is present"
    }
  ]
}
```

核心设计思想是：

- `SFT` 先教会模型“正确格式 + 正确字段 + 正确规则结果”。
- `DPO` 再让模型偏向 verifier 打分更高的输出。
- `RLVR` 最后直接把 verifier/composite reward 当训练信号继续优化。

## 2. 项目中的核心对象

理解这个仓库，先要把几种数据对象分清。

### 2.1 `input`

`input` 是模型看到的输入观察。

结构上主要包括：

- `sample_id`
- `form_type`
- `pdf_page`
- `ocr_tokens`

其中最关键的是 `ocr_tokens`。  
模型不是直接看 PDF 图片，而是看 OCR token 列表。

定义来源：

- [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py)
- [schema.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/schema.py)

示例：

```json
{
  "sample_id": "template_a_00000",
  "form_type": "insurance_application_form",
  "pdf_page": 1,
  "ocr_tokens": [
    {"text": "投保人姓名", "bbox": [16, 24, 106, 42], "page": 1},
    {"text": "王敏", "bbox": [126, 24, 276, 42], "page": 1},
    {"text": "出生日期", "bbox": [16, 160, 106, 178], "page": 1},
    {"text": "1990-01-O1", "bbox": [126, 160, 276, 178], "page": 1}
  ]
}
```

### 2.2 `reference`

`reference` 是 gold 标注答案。

包含两部分：

- `fields`
- `validations`

示例：

```json
{
  "sample_id": "template_a_00000",
  "fields": {
    "policyholder_name": "王敏",
    "policyholder_id_number": "44010119900101120X",
    "insured_birth_date": "1990-01-01",
    "payment_mode": "annual"
  },
  "validations": [
    {
      "rule_id": "required.policyholder_name",
      "status": "pass",
      "message": "policyholder_name is present"
    },
    {
      "rule_id": "consistency.birth_date_vs_id_number",
      "status": "pass",
      "message": "birth date matches id number"
    }
  ]
}
```

### 2.3 `prediction`

`prediction` 是模型实际生成的 JSON 结果。

仓库会尽量把模型文本解析成：

- `sample_id`
- `fields`
- `validations`

解析逻辑在：

- [predictions.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/predictions.py)

如果模型输出不是合法 JSON，会退化成空结构：

```json
{
  "sample_id": "xxx",
  "fields": {},
  "validations": []
}
```

### 2.4 `candidate`

`candidate` 是 baseline 推理阶段的多候选输出之一。

每个样本会通过 `SGLang` 采样出 `n` 个候选。当前默认是 `4`。

结构上包括：

- `candidate_id`
- `sample_id`
- `prediction`
- `raw_text`
- `generation_config`

代码在：

- [candidates.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/inference/candidates.py)

### 2.5 `preference`

`preference` 是 DPO 用的偏好对。

一条 preference 样本包含：

- 同一个 `sample_id`
- `input`
- `reference`
- `chosen`
- `rejected`
- `chosen_reward`
- `rejected_reward`
- `reward_margin`

代码在：

- [preferences.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/preferences.py)

### 2.6 `manifest`

`manifest.json` 是每个训练阶段的“训练声明”。

它描述：

- 用哪个 base model
- 用哪份 train data
- 输出到哪里
- adapter/precision 配置是什么
- trainer 参数是什么
- runtime backend 是什么

代码在：

- [manifests.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/manifests.py)

### 2.7 `runtime_plan`

`runtime_plan.json` 是从 `manifest.json` 推导出来的“实际执行计划”。

它描述：

- 真正要调用哪个 python/module
- 数据是否被 staged / materialized
- 最终命令行是什么
- 运行目录在哪里

代码在：

- [runtime.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/runtime.py)

## 3. 原始数据是怎么生成出来的

当前仓库默认并不是读取真实业务标注集，而是先用 synthetic 数据生成器构造样本。

入口：

- [scripts/generate_sft_dataset.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/generate_sft_dataset.py)

实现：

- [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py)

生成逻辑大致是：

1. 先构造一份结构化真值 `fields`。
2. 基于这些字段生成 OCR token。
3. 在 OCR token 里注入噪声。
4. 基于真值字段自动生成 `validations`。
5. 再根据任务类型导出成：
   - `SFT_gold`
   - `SFT_silver`
   - `RL_prompt_only`

### 3.1 synthetic 样本里包含哪些字段

当前字段主要包括：

- `policyholder_name`
- `policyholder_gender`
- `policyholder_id_number`
- `policyholder_phone`
- `policyholder_address`
- `insured_name`
- `insured_gender`
- `insured_id_number`
- `insured_birth_date`
- `relation_policyholder_to_insured`
- `product_name`
- `coverage_amount`
- `currency`
- `payment_mode`
- `payment_period_years`
- `beneficiary_name`
- `beneficiary_ratio`
- `signature_present`
- `application_date`
- `checkboxes`
- 条件字段 `auto_debit_account`

见：

- [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py#L178)

### 3.2 OCR 噪声是怎么设计的

当前 synthetic 里会模拟：

- 字符替换，比如 `0 -> O`
- 中文近形错字，比如 `保 -> 堡`
- 缺失字段
- 涂抹 / 遮挡
- checkbox 冲突

这意味着模型不是在学一个“干净表单抽取”问题，而是在学一个：

> OCR 噪声下的结构化抽取 + 规则判断问题

### 3.3 `SFT_gold` 和 `RL_prompt_only` 的区别

`SFT_gold`：

- 有 `input`
- 有 `reference`
- 用于监督训练

`RL_prompt_only`：

- 一定有 `input`
- 默认可以没有 `reference`
- 用于 RL 阶段让模型自由生成，再由 reward function 打分

在当前 synthetic 代码里，`task_type != RL_prompt_only` 才会写入 `reference`，见：

- [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py#L383)

## 4. SFT 阶段到底在训练什么

### 4.1 SFT 的训练目标

SFT 不是训练“从 OCR token 输出字段字典”这么简单。

它训练的是：

> 给定 OCR token 和任务说明，输出满足 schema 的完整 JSON，其中同时包含 `fields` 和 `validations`。

也就是说，SFT 的监督目标同时覆盖：

- 正确字段抽取
- 正确标准化
- 正确规则结果
- 正确 JSON 格式

### 4.2 SFT 训练数据是怎么从原始 JSON 变成 `messages` 的

SFT 训练语料构造在：

- [corpus.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/corpus.py#L21)

每条 `SFT_gold` 样本会被转换成：

```json
{
  "task_type": "SFT_gold",
  "stage": "phase_a_sft",
  "sample_id": "template_a_00000",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}
```

### 4.3 `system prompt` 是什么

定义在：

- [prompting.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/prompting.py#L8)

当前内容是：

```text
你是保险投保单结构化抽取与规则校验助手。请仅输出 JSON，对 OCR 结果做字段抽取并返回 validations。
```

### 4.4 `user prompt` 用了原始 JSON 的哪些部分

`user prompt` 只使用 `input` 里的信息，不把 `reference` 直接喂给模型。

具体使用：

- `sample_id`
- `form_type`
- `pdf_page`
- schema hint
- 所有 `ocr_tokens`

定义在：

- [prompting.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/prompting.py#L14)

所以，SFT 的 `user prompt` 本质上是：

> “这里是一页表单的 OCR token，请按这个 schema 输出 JSON。”

### 4.5 `assistant answer` 用了原始 JSON 的哪些部分

`assistant` 目标答案直接来自 `reference`，但只保留：

- `sample_id`
- `fields`
- `validations`

定义在：

- [prompting.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/prompting.py#L37)

所以 SFT 本质上在做：

```text
(system prompt)
+ (OCR token prompt)
-> (gold JSON answer)
```

### 4.6 SFT 的训练实现

SFT 的执行后端是 `transformers.Trainer`。

代码在：

- [trl_sft.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/trl_sft.py)

关键点：

- 把 `messages` 渲染成 chat text
- tokenizer 做截断
- `labels = input_ids`
- 用自回归语言模型做监督学习

也就是说，SFT 阶段是标准的 causal LM supervision。

### 4.7 SFT 的模型加载方式

当前默认训练配置是：

- `QLoRA`
- `4bit nf4`
- `bfloat16` compute dtype
- `gradient_checkpointing=true`

配置来源：

- [experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml)
- [finetune.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/finetune.py)

这意味着：

- 不是全参数微调
- 也不是全精度训练
- 是面向 4090/5090 的轻量 LoRA/QLoRA 路线

## 5. DPO 阶段到底在训练什么

### 5.1 DPO 的输入从哪里来

DPO 依赖两样东西：

1. baseline 候选输出
2. gold reference

在 pipeline 里它会：

- 先从 baseline 阶段拿 `candidates.jsonl`
- 再用 verifier reward 对候选排序
- 组装成 `chosen/rejected` preference pair

代码在：

- [stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py#L132)
- [preferences.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/preferences.py)

### 5.2 baseline 候选是怎么生成的

baseline 阶段会：

1. 读取 `sft_gold_path`
2. 取出每个样本的 `input`
3. 用 `SGLang` 请求 `/chat/completions`
4. 每条样本生成 `n=4` 个候选

实现：

- [stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py#L43)
- [candidates.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/inference/candidates.py)

### 5.3 为什么 `prepare-only` 也会占 GPU

这是很多人第一次看这个仓库最容易误解的点。

`prepare-only` 只表示：

- 不执行 SFT/DPO/RL 的训练命令

它 **不表示跳过 baseline 推理**。

代码证据：

- `--prepare-only` 只是把 `execution.prepare_only=True` 和 `execution.execute_training=False`，见 [runner.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/runner.py#L69)
- stage 顺序仍包含 `baseline`，见 [runner.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/runner.py#L84)
- baseline stage 会真实调用 `SGLang` 生成候选，见 [stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py#L43)

所以如果你在 `prepare-only` 时已经启动了 `SGLang`，显存大头通常会落在 `SGLang` 上。

### 5.4 DPO 的“偏好”是怎么定的

每个 candidate 会被：

1. 跑 verifier suite
2. 计算 reward
3. 再算评测指标
4. 按 reward 排序

默认 reward 由这些 verifier component 组成：

- `schema_reward`
- `field_match_reward`
- `normalization_reward`
- `cross_field_consistency_reward`
- `checkbox_logic_reward`
- `ocr_robustness_reward`

定义在：

- [compose.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/rewards/compose.py)
- [verifiers/__init__.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/verifiers/__init__.py)

### 5.5 DPO 样本长什么样

一条 DPO 样本大致是：

```json
{
  "task_type": "DPO_preference",
  "sample_id": "template_a_00000",
  "input": {...},
  "reference": {...},
  "chosen": {
    "prediction": {...},
    "metrics": {...},
    "verifier_results": [...]
  },
  "rejected": {
    "prediction": {...},
    "metrics": {...},
    "verifier_results": [...]
  },
  "chosen_reward": {...},
  "rejected_reward": {...},
  "reward_margin": 0.23
}
```

### 5.6 DPO 真正训练时喂给模型的是什么

真正给 TRL DPOTrainer 的数据又会再变一层。

在：

- [trl_dpo.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/trl_dpo.py)

会被转换成：

- `prompt`
- `chosen`
- `rejected`
- 一些 metadata

其中：

- `prompt` = `system_prompt + user_prompt`
- `chosen` = chosen candidate 的 JSON 文本
- `rejected` = rejected candidate 的 JSON 文本

所以 DPO 学的是：

> 在同一个 OCR prompt 下，更偏向 verifier 分数高的答案，而不是 verifier 分数低的答案。

## 6. RL 阶段到底在优化什么

### 6.1 RL 阶段输入是什么

RL 使用的是 `RL_prompt_only` 数据。

在 pipeline 中对应：

- [stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py#L189)

它会将每条样本转换成：

- `prompt`
- `system_prompt`
- `reward_profile`
- 可选 `reference`

构造逻辑在：

- [corpus.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/corpus.py#L78)

### 6.2 RL 的 reward 从哪里来

当前 RL reward 不是人类偏好打分，也不是单纯 token-level 奖励。

它来自：

1. 模型输出一段文本
2. `parse_prediction_text()` 把它解析成 JSON
3. `run_verifier_suite()` 跑各个 verifier
4. `score_verifier_results()` 聚合成最终 reward

代码在：

- [verl_reward.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/verl_reward.py)

这就是所谓的 `RLVR`：

> Reinforcement Learning with Verifiable Rewards

### 6.3 RL 训练后端是什么

当前 RL 阶段使用：

- `verl`
- rollout engine = `sglang`

配置来源：

- [experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml)
- [runtime.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/runtime.py)

也就是说，RL 阶段实际上是：

- 一个训练端
- 一个 rollout 推理端
- 一个 reward 计算函数

共同组成的系统。

这也是为什么 RL 阶段通常是整个 pipeline 里资源要求最高、最复杂、最容易踩坑的一段。

## 7. 评测指标是什么

当前评测核心指标在：

- [metrics.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/evaluation/metrics.py)

主要包括：

- `field_precision`
- `field_recall`
- `field_f1`
- `form_exact_match`
- `rule_pass_rate`
- `validation_match_rate`
- `invalid_json_rate`

这些指标反映了三类能力：

1. 字段值对不对。
2. 规则状态对不对。
3. 输出 JSON 是否合法。

## 8. pipeline 的每个阶段到底做什么

当前 AutoDL pipeline 配置见：

- [pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml)

默认开启四段：

1. `baseline`
2. `phase_a_sft`
3. `phase_b_dpo`
4. `phase_c_grpo`

### 8.1 `baseline`

输入：

- `sft_gold.jsonl`
- `VERIDOC_MODEL_REF`
- `SGLang` 服务

处理：

- 对每条样本生成 4 个候选
- 取第 1 个候选作为 baseline prediction
- 对 baseline prediction 做评测

输出：

- `baseline/candidates.jsonl`
- `baseline/predictions.jsonl`
- `baseline/report.json`
- `baseline/cases.jsonl`

### 8.2 `phase_a_sft`

输入：

- `sft_gold.jsonl`
- base model

处理：

- 构造 `messages` 训练语料
- 写 manifest
- 写 runtime plan
- 真跑训练时输出 checkpoint

输出：

- `phase_a_sft/train.jsonl`
- `phase_a_sft/manifest.json`
- `phase_a_sft/runtime_plan.json`
- `phase_a_sft/launch.sh`
- 真训练后还有 `phase_a_sft/checkpoints/`

### 8.3 `phase_b_dpo`

输入：

- `baseline/candidates.jsonl`
- `sft_gold.jsonl`
- base model = SFT checkpoint

处理：

- 用 verifier reward 选择 chosen / rejected
- 构造 DPO corpus
- 写 manifest / runtime plan

输出：

- `phase_b_dpo/preferences.jsonl`
- `phase_b_dpo/train.jsonl`
- `phase_b_dpo/manifest.json`
- `phase_b_dpo/runtime_plan.json`
- `phase_b_dpo/launch.sh`
- 真训练后还有 `phase_b_dpo/checkpoints/`

### 8.4 `phase_c_grpo`

输入：

- `rl_prompt_only.jsonl`
- base model = SFT checkpoint

处理：

- 构造 RL prompt-only corpus
- 物化为 verl 所需数据格式
- 绑定 custom reward function
- 写 runtime plan

输出：

- `phase_c_grpo/train.jsonl`
- `phase_c_grpo/manifest.json`
- `phase_c_grpo/runtime_plan.json`
- `phase_c_grpo/launch.sh`
- `phase_c_grpo/data/*.parquet`
- 真训练后还有 `phase_c_grpo/checkpoints/`

## 9. `prepare-only` 会产生什么文件

这是理解整个系统非常重要的一步。

执行：

```bash
python scripts/run_pipeline.py \
  --spec-path configs/pipeline.autodl.qwen3_1p7.yaml \
  --prepare-only
```

之后，通常会在：

```text
${VERIDOC_OUTPUT_ROOT}/qwen3_1p7_autodl
```

看到下面这些文件：

- `state.json`
- `spec.snapshot.json`
- `baseline/`
- `phase_a_sft/`
- `phase_b_dpo/`
- `phase_c_grpo/`

这里最重要的是：

- `state.json`
  - pipeline 当前状态
- `manifest.json`
  - 每个训练阶段的“配置声明”
- `runtime_plan.json`
  - 每个训练阶段的“实际执行计划”
- `launch.sh`
  - 最终可执行命令包装

也就是说，`prepare-only` 的核心作用是：

> 把未来真正训练要用的所有中间配置和中间语料先生成出来，让你先检查。

## 10. 文件映射总表

### 10.1 顶层配置与入口

| 类型 | 文件 |
|---|---|
| AutoDL 环境变量模板 | [configs/autodl.env.example](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/autodl.env.example) |
| AutoDL pipeline spec | [configs/pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml) |
| 实验矩阵 | [configs/experiment_matrix.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/experiment_matrix.yaml) |
| 环境安装脚本 | [scripts/bootstrap_autodl_envs.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/bootstrap_autodl_envs.sh) |
| 启动 SGLang | [scripts/start_sglang_server.sh](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/start_sglang_server.sh) |
| 主 pipeline 入口 | [scripts/run_pipeline.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/run_pipeline.py) |

### 10.2 数据生成

| 环节 | 文件 |
|---|---|
| synthetic 数据入口 | [scripts/generate_sft_dataset.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/scripts/generate_sft_dataset.py) |
| synthetic 数据实现 | [src/veridoc_rl/data/synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py) |
| schema 定义 | [src/veridoc_rl/schema.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/schema.py) |

### 10.3 Prompt 与训练语料

| 环节 | 文件 |
|---|---|
| prompt 构造 | [src/veridoc_rl/training/prompting.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/prompting.py) |
| SFT/DPO/RL corpus 构造 | [src/veridoc_rl/training/corpus.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/corpus.py) |

### 10.4 推理与候选生成

| 环节 | 文件 |
|---|---|
| baseline 候选生成 | [src/veridoc_rl/inference/candidates.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/inference/candidates.py) |
| checkpoint 推理 | [src/veridoc_rl/inference/runner.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/inference/runner.py) |
| 预测解析 | [src/veridoc_rl/predictions.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/predictions.py) |

### 10.5 奖励与验证

| 环节 | 文件 |
|---|---|
| verifier suite 入口 | [src/veridoc_rl/verifiers/__init__.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/verifiers/__init__.py) |
| verifier base | [src/veridoc_rl/verifiers/base.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/verifiers/base.py) |
| reward 聚合 | [src/veridoc_rl/rewards/compose.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/rewards/compose.py) |
| DPO preference 构造 | [src/veridoc_rl/data/preferences.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/preferences.py) |
| RL reward function | [src/veridoc_rl/training/verl_reward.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/verl_reward.py) |

### 10.6 训练执行

| 环节 | 文件 |
|---|---|
| manifest 构造 | [src/veridoc_rl/training/manifests.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/manifests.py) |
| runtime plan 构造 | [src/veridoc_rl/training/runtime.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/runtime.py) |
| SFT 训练 | [src/veridoc_rl/training/trl_sft.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/trl_sft.py) |
| DPO 训练 | [src/veridoc_rl/training/trl_dpo.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/trl_dpo.py) |
| LoRA / QLoRA / 4bit 加载 | [src/veridoc_rl/training/finetune.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/finetune.py) |

### 10.7 编排与状态管理

| 环节 | 文件 |
|---|---|
| pipeline 入口 | [src/veridoc_rl/orchestration/runner.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/runner.py) |
| stage 执行逻辑 | [src/veridoc_rl/orchestration/stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py) |
| pipeline 路径规则 | [src/veridoc_rl/orchestration/paths.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/paths.py) |
| state 管理 | [src/veridoc_rl/orchestration/state.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/state.py) |
| spec 解析 | [src/veridoc_rl/orchestration/spec.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/spec.py) |

## 11. 数据流流程图

下面这张图是整个 pipeline 的文字版数据流。

```text
                              +---------------------------+
                              | configs/autodl.env.example|
                              +-------------+-------------+
                                            |
                                            v
                              +---------------------------+
                              | configs/pipeline.autodl   |
                              +-------------+-------------+
                                            |
                                            v
                              +---------------------------+
                              | scripts/run_pipeline.py   |
                              +-------------+-------------+
                                            |
              +-----------------------------+-----------------------------+
              |                             |                             |
              v                             v                             v
   +--------------------+      +-------------------------+      +----------------------+
   | SFT_gold.jsonl     |      | RL_prompt_only.jsonl    |      | experiment_matrix    |
   | generate_sft_dataset|      | generate_sft_dataset    |      | yaml                 |
   +----------+---------+      +------------+------------+      +----------+-----------+
              |                               |                              |
              |                               |                              |
              |                               |                              |
              v                               |                              |
   +--------------------+                     |                              |
   | baseline stage     |                     |                              |
   | SGLang generate n  |                     |                              |
   +----------+---------+                     |                              |
              |                               |                              |
              v                               |                              |
   +--------------------+                     |                              |
   | candidates.jsonl   |                     |                              |
   +----------+---------+                     |                              |
              |                               |                              |
              +------------------+            |                              |
                                 |            |                              |
                                 v            v                              v
                        +-----------------------------------------------------------+
                        | orchestration/stages.py                                   |
                        +----------------+------------------+------------------------+
                                         |                  |
                                         |                  |
                                         v                  v
                             +------------------+   +----------------------+
                             | phase_a_sft      |   | phase_c_grpo         |
                             | prepare_sft_corpus|   | prepare_rl_corpus    |
                             +--------+---------+   +----------+-----------+
                                      |                        |
                                      v                        v
                             +------------------+   +----------------------+
                             | train.jsonl      |   | train.jsonl          |
                             +--------+---------+   +----------+-----------+
                                      |                        |
                                      |                        |
                                      v                        v
                             +------------------+   +----------------------+
                             | manifest.json    |   | manifest.json        |
                             | runtime_plan.json|   | runtime_plan.json    |
                             | launch.sh        |   | launch.sh            |
                             +--------+---------+   +----------+-----------+
                                      |                        |
                                      |                        |
                                      v                        v
                             +------------------+   +----------------------+
                             | SFT checkpoint    |   | RL checkpoint        |
                             +--------+---------+   +----------+-----------+
                                      |
                                      v
                             +------------------+
                             | phase_b_dpo      |
                             | preference build |
                             +--------+---------+
                                      |
                                      v
                             +------------------+
                             | preferences.jsonl|
                             | train.jsonl      |
                             | manifest/runtime |
                             +--------+---------+
                                      |
                                      v
                             +------------------+
                             | DPO checkpoint    |
                             +------------------+
```

## 12. 资源消耗预估

这一节是 **工程估算**，不是代码里硬编码的精确保证。

影响资源消耗的关键因素包括：

- 模型本体大小
- dtype
- QLoRA / 4bit 是否启用
- prompt 长度
- completion 长度
- candidate 数量 `n`
- rollout 配置
- `SGLang` / `verl` 的显存预留策略

### 12.1 baseline / `prepare-only` 阶段

当前配置：

- baseline model: `Qwen/Qwen3-1.7B`
- backend: `sglang`
- `candidate_count=4`
- `max_new_tokens=1024`

估算：

- GPU 显存：约 `12GB - 24GB`
- CPU 内存：约 `4GB - 12GB`
- 磁盘：
  - 模型缓存若在线拉取，通常 `几 GB`
  - baseline 输出文件较小，通常 `MB` 级

说明：

- 这里显存大头通常不是 1.7B 权重本身，而是 `SGLang` 的 KV cache 与显存池预留。
- 所以你看到 `21GB` 并不反常，尤其是 `n=4`、输出长度较长时。

### 12.2 SFT 阶段

当前配置：

- `QLoRA`
- `load_in_4bit=true`
- `torch_dtype=bfloat16`
- `gradient_checkpointing=true`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `max_length=3072`

估算：

- GPU 显存：约 `10GB - 18GB`
- CPU 内存：约 `8GB - 20GB`
- 磁盘：
  - 训练数据很小，通常 `MB` 级
  - checkpoint 取决于保存步数，通常 `GB` 级以内

说明：

- 对 1.7B + QLoRA 来说，SFT 通常不会比 RL 阶段更重。
- 真正压力主要来自较长序列长度 `3072`。

### 12.3 DPO 阶段

当前配置：

- base model = SFT checkpoint
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `max_length=3072`
- `max_prompt_length=2048`
- `max_completion_length=1024`

估算：

- GPU 显存：约 `14GB - 24GB`
- CPU 内存：约 `8GB - 24GB`
- 磁盘：
  - preference 数据通常比 SFT 稍大
  - checkpoint 仍为 `GB` 级

说明：

- DPO 往往比 SFT 更重，因为一个样本里同时涉及 `prompt + chosen + rejected`。

### 12.4 RL 阶段

当前配置：

- backend: `verl`
- rollout engine: `sglang`
- `rollout_n=4`
- `rollout_gpu_memory_utilization=0.5`
- 单机单卡

估算：

- GPU 显存：约 `18GB - 32GB+`
- CPU 内存：约 `12GB - 32GB`
- 磁盘：
  - RL 运行目录、log、checkpoint、stage 数据会明显增多
  - 建议预留 `10GB - 30GB+`

说明：

- RL 是整个 pipeline 里最重的一段。
- 原因不是模型本体突然变大，而是：
  - 训练 actor
  - rollout
  - reward 计算
  - 可能还伴随 reference/log-prob 相关开销

### 12.5 AutoDL 机器的实践建议

如果你要稳定跑通全链路，建议：

- GPU：4090 或 5090
- 系统内存：越大越稳，至少别太小
- 磁盘分工：
  - `/root/autodl-fs` 放代码、模型、HF cache
  - `/root/autodl-tmp` 放 venv、cache、输出、checkpoint

## 13. 你可以如何分阶段理解这个项目

如果你想真正把整个项目吃透，推荐按这个顺序：

1. 先看 synthetic 数据长什么样。
2. 再看 SFT 是怎么把 `input/reference` 变成 `messages` 的。
3. 再看 baseline 是怎么通过 `SGLang` 生成 candidates 的。
4. 再看 verifier reward 如何把 candidates 排序成 preference。
5. 再看 DPO 样本怎样从 preference 变成 `prompt/chosen/rejected`。
6. 最后看 RL 的 `verl_reward.py`，理解 verifier reward 怎样进入强化学习闭环。

## 14. 最重要的认知总结

如果你只记住三点，记这三点就够了。

第一：

> 这个项目的目标输出从头到尾都不是“字段字典”，而是“字段 + 规则校验”的统一 JSON。

第二：

> `prepare-only` 依然会执行 baseline 推理，所以只要你开了 `SGLang`，就仍然可能看到明显 GPU 显存占用。

第三：

> `SFT -> DPO -> RL` 不是三个毫无关系的训练，而是同一个任务目标的三次逐步强化：
> `SFT` 学格式和基本能力，`DPO` 学 verifier 偏好，`RL` 直接优化 verifier reward。
