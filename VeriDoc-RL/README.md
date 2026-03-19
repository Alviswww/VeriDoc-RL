# VeriDoc-RL

VeriDoc-RL 是一个面向制式投保单的 verifier-guided post-training 项目。它把 `OCR token 序列 -> 结构化字段抽取 -> 规则校验 -> verifier reward` 串成统一闭环，用同一套 schema、规则、verifier 和 reward 支持 `SFT / DPO / RLVR` 三条路线。

这份 README 是仓库的主接手文档。后续开新线程时，优先读本文件即可建立对仓库的整体认知，通常不需要再把 `docs/`、`scripts/`、`src/` 整体重新扫一遍。

## 1. 仓库定位

一句话：

`输入投保单 OCR/版面结果，输出 fields + validations，并比较 SFT / DPO / RLVR 在字段准确率、规则通过率和 OCR 噪声鲁棒性上的差异。`

项目强调的不是“再做一个表单抽取任务”，而是：

- 把制式文档抽取建模为 `结构化生成 + 规则校验` 的联合输出问题。
- 把业务约束改写成可程序化 verifier。
- 用同一套 verifier 同时服务于：
  - Phase A 评测
  - DPO preference 构造
  - RLVR composite reward
- 用 bucket 分析、error taxonomy 和 case export 做闭环验证。

## 2. 当前实现状态

### 已完成

- 统一输入输出 schema
- synthetic 投保单样本生成
- 基于规则目录的 reference validations 构造
- 六类 verifier
- reward profile / ablation
- Phase A 单报告评测
- 多报告对比、Markdown 摘要与 SVG 图表导出
- experiment matrix 解析与实验计划展开
- Phase A / B / C 训练语料准备脚手架
- Phase B / C 训练 manifest 生成
- Phase B TRL-backed DPO runtime launch plan
- Phase C `verl` runtime launch plan 与 verifier reward bridge

### 当前仍然是脚手架的部分

- 还没有接入真正的分布式训练编排、checkpoint 恢复和训练日志归档
- 当前 Phase B DPO runtime 会生成可执行 launch plan，并通过仓库内 `TRL` runner 消费本地 JSONL；真正执行仍依赖你本地已安装训练依赖与可用 GPU 环境
- 当前 `verl` runtime adapter 已能为 `phase_c_grpo / phase_c_rloo` 生成真实 launch plan，但仍依赖你本地已安装 `verl` 与可用 GPU 环境
- 还没有 candidate 采样器、online rollout executor、checkpoint 管理和训练日志聚合
- 当前没有独立的 reward model 训练 pipeline；RL 仍默认使用 verifier-based reward，而不是可训练 RM

也就是说：仓库已经具备数据、verifier、reward、评测、训练语料准备、训练配置生成，以及 Phase B / C 的运行时桥接；真正还缺的是更完整的训练编排层与独立 reward model 链路，而不是底层 reward/评测骨架。

## 3. 目录与根路径说明

有一个容易踩坑的点：

- git 根目录是外层：`/home/alvis/projects/llm-study/VeriDoc-RL`
- Python 项目根目录是内层：`/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL`

下面 README 里的所有命令，默认都在内层项目根目录执行：

```bash
cd /home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL
```

## 4. 总体架构

从数据到训练的主数据流如下：

1. `src/veridoc_rl/data/synthetic.py`
   - 生成 synthetic `input/reference/metadata`
   - 输出 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
2. `src/veridoc_rl/verifiers/`
   - 对预测结果做 schema、字段、标准化、跨字段、checkbox、OCR robustness 验证
3. `src/veridoc_rl/rewards/`
   - 把 verifier score 组合成 composite reward
4. `src/veridoc_rl/data/preferences.py`
   - 用 verifier score 给多候选 prediction 排序
   - 构造 `DPO_preference`
5. `src/veridoc_rl/evaluation/reporting.py`
   - 做单次 Phase A 评测
   - 输出 overall、bucket、taxonomy、failure cases
6. `src/veridoc_rl/evaluation/comparison.py`
   - 对多份评测报告做横向对比
   - 产出 Markdown 和 SVG 图表
7. `src/veridoc_rl/training/corpus.py`
   - 把原始 JSONL 转成训练语料
   - 支持 Phase A SFT、Phase B DPO、Phase C RLVR
8. `src/veridoc_rl/training/manifests.py`
   - 从 experiment matrix 和训练语料路径生成训练 manifest bundle
9. `src/veridoc_rl/training/runtime.py`
   - 从 `manifest.json` 生成 `runtime_plan.json` 和 `launch.sh`
   - 把 `phase_b_dpo` 桥接到仓库内 `TRL` DPO runner
   - 把 `phase_c_grpo / phase_c_rloo` 桥接到 `verl.trainer.main_ppo`
10. `src/veridoc_rl/training/trl_dpo.py`
   - 提供 TRL-backed DPO runner
   - 负责本地 JSONL -> TRL dataset rows 的转换与训练执行
11. `src/veridoc_rl/training/verl_reward.py`
   - 提供 `verl` 可调用的 verifier-based `compute_score`

## 5. 数据契约

### 5.1 输入 schema

每条输入样本的核心结构：

```json
{
  "sample_id": "template_a_00001",
  "form_type": "insurance_application_form",
  "pdf_page": 1,
  "ocr_tokens": [
    {
      "text": "投保人姓名",
      "bbox": [10, 20, 60, 40],
      "page": 1
    }
  ]
}
```

### 5.2 输出 schema

模型必须输出：

```json
{
  "sample_id": "template_a_00001",
  "fields": {
    "policyholder_name": "张三"
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

### 5.3 训练前数据类型

仓库当前支持四类上游 JSONL：

- `SFT_gold`
- `SFT_silver`
- `RL_prompt_only`
- `DPO_preference`

### 5.4 训练语料类型

`src/veridoc_rl/training/corpus.py` 会把它们转成三类下游训练语料：

- `phase_a_sft`
  - `messages = [system, user, assistant]`
- `phase_b_dpo`
  - `system_prompt + prompt + chosen + rejected`
- `phase_c_rlvr`
  - `system_prompt + prompt + reward_profile + metadata`

## 6. 字段、规则、bucket 与 reward

### 6.1 字段范围

第一版字段覆盖以下方向：

- 投保人/被保人基本信息
- 证件号、手机号、地址
- 险种、保额、缴费方式、缴费年限
- 受益人信息与比例
- 勾选项
- 签字/日期类字段

实际 synthetic generator 当前稳定覆盖的代表字段包括：

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
- `auto_debit_account`

### 6.2 规则目录

规则定义集中在 `src/veridoc_rl/rules.py`，文档版目录在 `docs/rules_catalog.md`。

当前规则包括三大类：

- required
  - `required.policyholder_name`
  - `required.policyholder_id_number`
  - `required.insured_name`
- format
  - `format.policyholder_phone`
  - `format.policyholder_id_number`
  - `format.application_date`
- consistency / checkbox
  - `consistency.birth_date_vs_id_number`
  - `consistency.beneficiary_ratio_sum`
  - `consistency.policyholder_insured_relation`
  - `consistency.product_payment_combo`
  - `checkbox.payment_mode_exclusive`
  - `checkbox.auto_debit_requires_account`

### 6.3 Bucket 维度

仓库统一按四个维度做分桶：

- `template_family`
- `ocr_noise_level`
- `hard_case_type`
- `rule_complexity`

### 6.4 Reward 组件

当前 reward 组件：

- `schema_reward`
- `field_match_reward`
- `normalization_reward`
- `cross_field_consistency_reward`
- `checkbox_logic_reward`
- `ocr_robustness_reward`

支持的 reward profile：

- `default`
- `rlvr`
- `rlvr_without_cross_field_consistency`
- `rlvr_without_checkbox_logic`

## 7. 代码模块索引

### 7.1 `src/veridoc_rl/schema.py`

统一 dataclass schema：

- `OCRToken`
- `FormInput`
- `ValidationResult`
- `FormOutput`
- `validate_prediction_payload`

### 7.2 `src/veridoc_rl/normalizers.py`

字段标准化工具：

- `normalize_phone`
- `normalize_id_number`
- `normalize_date`
- `normalize_amount`
- `normalize_checkbox_value`
- `normalize_known_field`

### 7.3 `src/veridoc_rl/data/synthetic.py`

synthetic 数据生成器：

- `SyntheticFormGenerator`
- `build_validations`
- `apply_ocr_noise`
- `build_training_record`
- CLI: `generate_sft_dataset.py`

### 7.4 `src/veridoc_rl/verifiers/`

当前默认 verifier suite 顺序：

1. `SchemaVerifier`
2. `FieldMatchVerifier`
3. `NormalizationVerifier`
4. `CrossFieldConsistencyVerifier`
5. `CheckboxLogicVerifier`
6. `OCRRobustnessVerifier`

### 7.5 `src/veridoc_rl/rewards/`

负责：

- reward 权重定义
- ablation profile
- verifier result -> total reward 聚合

### 7.6 `src/veridoc_rl/evaluation/`

分两层：

- `metrics.py`
  - 单样本字段/表单/规则指标
- `reporting.py`
  - 单报告汇总、bucket、taxonomy、failure cases
- `comparison.py`
  - 多报告对比、Markdown、SVG

### 7.7 `src/veridoc_rl/experiments/`

负责消费 `configs/experiment_matrix.yaml`：

- 解析 experiment matrix
- 展开实验队列
- 输出 JSON / Markdown 实验计划

### 7.8 `src/veridoc_rl/training/`

这是本轮新补的训练侧脚手架：

- `prompting.py`
  - 统一 system prompt
  - OCR token -> user prompt
  - reference output -> assistant response
- `corpus.py`
  - 原始 JSONL -> 训练语料 JSONL
- `manifests.py`
  - 根据 experiment matrix 生成 `phase_b_dpo` / `phase_c_grpo` / `phase_c_rloo`
  - 输出 `manifest.json`、`README.md`、`verl.yaml`
- `runtime.py`
  - 读取 manifest
  - 生成 launch plan
  - 为 RL phase 生成 `launch.sh` 和 `runtime_plan.json`
- `verl_reward.py`
  - 提供 `custom_reward_function.path` 指向的 reward bridge

## 8. `experiment_matrix.yaml` 说明

位置：

```text
configs/experiment_matrix.yaml
```

当前它描述五类内容：

- project 目标
- base model 分层
- 数据 bucket 与 error taxonomy
- training stage 定义
- training runtime 默认项
- reward component
- evaluation 关心的 primary metrics 与 ablations

训练 runtime 现在也在这里维护，包含：

- `backend`
- `prompt_template`
- `phases.phase_b_dpo`
- `phases.phase_c_grpo`
- `phases.phase_c_rloo`

这使得后续接入真实 trainer 时，可以继续沿用同一份 matrix 作为“实验真相来源”。

## 9. 环境与安装

安装开发环境：

```bash
pip install -e .[dev]
```

开发依赖当前只有：

- `pytest`
- `ruff`
- `mypy`

如果要实际执行 Phase B DPO 训练，再安装训练依赖：

```bash
pip install -e .[train]
```

其中会引入：

- `trl`
- `transformers`
- `datasets`
- `accelerate`
- `peft`

## 10. 常用命令总表

### 10.1 运行测试

```bash
pytest
```

### 10.2 最小 smoke

```bash
veridoc-rl-smoke
```

也可以传 fixture 或 prediction：

```bash
veridoc-rl-smoke --fixture-path path/to/fixture.json
veridoc-rl-smoke --prediction-json '{"sample_id":"s1","fields":{},"validations":[]}'
```

### 10.3 生成 synthetic 数据

生成 SFT 数据：

```bash
python scripts/generate_sft_dataset.py \
  --count 8 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path outputs/sft_gold.jsonl
```

生成 RL prompt-only 数据：

```bash
python scripts/generate_sft_dataset.py \
  --count 8 \
  --seed 7 \
  --task-type RL_prompt_only \
  --output-path outputs/rl_prompt_only.jsonl
```

### 10.4 生成 preference 数据

```bash
python scripts/generate_preference_dataset.py \
  --reference-path outputs/reference.jsonl \
  --candidate-path outputs/candidates.jsonl \
  --output-path outputs/preferences.jsonl \
  --min-margin 0.05
```

### 10.5 运行单报告评测

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/reference.jsonl \
  --prediction-path outputs/prediction.jsonl \
  --report-path outputs/phase_a_report.json \
  --case-export-path outputs/phase_a_cases.jsonl \
  --failure-only
```

### 10.6 做 reward ablation

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/reference.jsonl \
  --prediction-path outputs/prediction.jsonl \
  --report-path outputs/phase_a_report.ablation.json \
  --reward-profile rlvr_without_checkbox_logic
```

### 10.7 生成实验计划

```bash
python scripts/generate_experiment_plan.py \
  --matrix-path configs/experiment_matrix.yaml \
  --output-path outputs/experiment_plan.json \
  --markdown-path outputs/experiment_plan.md
```

也可以直接用 console script：

```bash
veridoc-rl-generate-plan \
  --matrix-path configs/experiment_matrix.yaml \
  --output-path outputs/experiment_plan.json \
  --markdown-path outputs/experiment_plan.md
```

### 10.8 对比多份评测报告

```bash
python scripts/compare_phase_reports.py \
  --report sft=outputs/sft_report.json \
  --report dpo=outputs/dpo_report.json \
  --report rlvr=outputs/rlvr_report.json \
  --output-dir outputs/report_compare
```

输出目录会生成：

- `comparison.json`
- `comparison.md`
- `rule_pass_rate_comparison.svg`
- `ocr_noise_level_field_f1.svg`

### 10.9 准备训练语料

Phase A SFT 语料：

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/sft_gold.jsonl \
  --output-path outputs/train.phase_a_sft.jsonl \
  --stage phase_a_sft
```

Phase B DPO 语料：

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/preferences.jsonl \
  --output-path outputs/train.phase_b_dpo.jsonl \
  --stage phase_b_dpo
```

Phase C RLVR 语料：

```bash
python scripts/prepare_training_data.py \
  --input-path outputs/rl_prompt_only.jsonl \
  --output-path outputs/train.phase_c_rlvr.jsonl \
  --stage phase_c_rlvr \
  --reward-profile rlvr
```

console script 版本：

```bash
veridoc-rl-prepare-training \
  --input-path outputs/rl_prompt_only.jsonl \
  --output-path outputs/train.phase_c_rlvr.jsonl \
  --stage phase_c_rlvr \
  --reward-profile rlvr
```

### 10.10 生成 Phase B / Phase C 训练 manifest

```bash
python scripts/generate_training_manifests.py \
  --matrix-path configs/experiment_matrix.yaml \
  --train-data-path outputs/train.phase_b_dpo.jsonl \
  --output-dir outputs/training_bundle
```

如果要给 RL 阶段生成 bundle，也直接复用：

```bash
python scripts/generate_training_manifests.py \
  --matrix-path configs/experiment_matrix.yaml \
  --train-data-path outputs/train.phase_c_rlvr.jsonl \
  --output-dir outputs/training_bundle_rl
```

生成结果默认包含：

- `phase_b_dpo/manifest.json`
- `phase_b_dpo/README.md`
- `phase_b_dpo/verl.yaml`
- `phase_c_grpo/manifest.json`
- `phase_c_grpo/README.md`
- `phase_c_grpo/verl.yaml`
- `phase_c_rloo/manifest.json`
- `phase_c_rloo/README.md`
- `phase_c_rloo/verl.yaml`

注意：

- 这一步会生成 manifest bundle，但不会自动启动训练。
- `phase_b_dpo` 现在可以继续走下一步 `prepare_training_runtime.py`，生成 TRL-backed DPO launch plan。
- `phase_c_grpo / phase_c_rloo` 也继续走下一步 `prepare_training_runtime.py`，生成 `verl` launch plan。

### 10.11 为 Phase B / Phase C 准备 runtime

给 `phase_b_dpo` 生成 launch plan：

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle/phase_b_dpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_b_dpo
```

DPO 生成结果包含：

- `runtime_plan.json`
- `launch.sh`
- `dpo_config.json`
- `data/phase_b_dpo.train.jsonl`

DPO 运行逻辑：

- `phase_b_dpo` 会桥接到 `python -m veridoc_rl.training.trl_dpo --config-path ...`
- runtime 会把仓库内 `phase_b_dpo` 训练语料转成 TRL 可直接消费的 `prompt/chosen/rejected` JSONL
- 真正执行训练前需要先安装 `.[train]`

给 `phase_c_grpo` 生成 launch plan：

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_c_grpo
```

如果训练语料是 JSONL，并且你已经在训练环境安装了 `pyarrow`，可以让脚本顺手物化成 parquet：

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_c_grpo \
  --materialize-data
```

如果当前环境已经安装 `verl`，也可以直接执行：

```bash
python scripts/prepare_training_runtime.py \
  --manifest-path outputs/training_bundle_rl/phase_c_grpo/manifest.json \
  --run-dir outputs/runtime_runs/phase_c_grpo \
  --materialize-data \
  --execute
```

RL 生成结果包含：

- `runtime_plan.json`
- `launch.sh`
- `data/phase_c_grpo.train.parquet`，仅在需要且成功物化时生成

运行逻辑：

- `phase_c_grpo` / `phase_c_rloo` 会桥接到 `python -m verl.trainer.main_ppo`
- 通过 `algorithm.adv_estimator=grpo|rloo` 区分算法
- 通过 `custom_reward_function.path=.../verl_reward.py` 接入当前仓库 verifier reward
- `phase_c` 当前仍默认使用 verifier reward，而不是独立 reward model
- `python scripts/prepare_verl_runtime.py` 仍保留为兼容入口，但推荐统一使用 `prepare_training_runtime.py`

## 11. 推荐执行路径

如果从零开始走一遍最小闭环，建议顺序：

1. 安装依赖并跑 `pytest`
2. 跑 `veridoc-rl-smoke`
3. 生成 `SFT_gold` 或 `RL_prompt_only`
4. 准备 Phase A 训练语料
5. 跑 `run_phase_a_eval.py`
6. 对不同 reward profile 生成多份 report
7. 跑 `compare_phase_reports.py`
8. 构造 `DPO_preference`
9. 用 `prepare_training_data.py` 导出 `phase_b_dpo`
10. 用 `generate_training_manifests.py` 生成 manifest bundle
11. 用 `prepare_training_runtime.py` 为 `phase_b_dpo` 或 `phase_c_*` 生成 launch plan

## 12. 测试覆盖

当前测试关注：

- schema round-trip
- rule registry
- normalizer/verifier/reward
- synthetic generator
- preference 构造
- Phase A 评测
- experiment matrix 与多报告对比
- training prompt / corpus / manifest 脚手架

如果你改动以下模块，建议对应补测试：

- `data/` 变更：补 synthetic / preference / training corpus 测试
- `verifiers/` 变更：补 verifier 与 reward 测试
- `evaluation/` 变更：补 report / comparison 测试
- `training/` 变更：补 CLI 输出与 manifest 测试

## 13. 推荐开发流程

建议始终用 worktree 做新增模块开发：

```bash
cd /home/alvis/projects/llm-study/VeriDoc-RL
git worktree add /tmp/veridoc-rl-next -b feature_x
cd /tmp/veridoc-rl-next/VeriDoc-RL

pip install -e .[dev]
pytest

# 如果要执行 DPO 训练，再补训练依赖
pip install -e .[train]

cd /home/alvis/projects/llm-study/VeriDoc-RL
git merge --ff-only feature_x
git worktree remove /tmp/veridoc-rl-next
```

原因很简单：

- 主工作树保持干净
- 新线程接手时容易区分“主线代码”和“功能分支”
- 便于在合并前独立验证

## 14. 已知限制

- synthetic 数据仍然是小规模规则驱动，未覆盖真实保单复杂版式
- OCR robustness 目前通过 `perturbed_predictions` 近似，不是真实 online augmentation
- 当前 DPO runtime 通过仓库内 `TRL` runner 执行，尚未接入更完整的多机编排、resume 和日志聚合
- launch plan 默认假设 `verl` 使用 parquet 数据；若上游语料是 JSONL，实际执行前需要 `pyarrow`
- 当前没有独立的 reward model 训练 / 推理 pipeline；RL 继续默认走 verifier reward
- 还没有模型推理、candidate 采样和训练日志回流

## 15. 后续优先建议

如果继续开发，优先顺序建议如下：

1. 接入真实 candidate sampler
2. 增加 checkpoint 目录协议与实验日志聚合
3. 加 online rollout executor / candidate sampler
4. 单独补 reward model 数据、训练与 serving pipeline
5. 把 comparison 结果自动写入 experiment registry

## 16. 其他文档何时再看

如果只需要继续编码，通常看 README 就够。

只有在以下情况再读其他文档：

- 需要项目原始目标和排期：`docs/project_blueprint.md`
- 需要对外表述版本：`docs/resume_alignment.md`
- 需要规则目录全文：`docs/rules_catalog.md`

## 17. 简版结论

这个仓库现在已经不是单纯的 verifier demo，而是一个可继续往真实后训练系统推进的骨架：

- 数据生成已通
- verifier/reward 已通
- 评测与对比已通
- Phase B / C 训练语料已通
- Phase B 的 TRL-backed DPO runtime 已通
- Phase C 的 `verl` runtime launch adapter 已通

下一步真正缺的是更完整的训练编排层和独立 reward model，而不是再补新的 schema。
