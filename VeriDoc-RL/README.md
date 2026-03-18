# VeriDoc-RL

VeriDoc-RL 是一个面向制式投保单的 RLVR 项目。项目的核心是把 `OCR 后的表单结果 -> 结构化抽取 -> 规则校验 -> verifier-guided reward` 串成完整的后训练闭环。

## 一句话定位

`输入投保单 OCR/版面结果，输出结构化字段和校验报告，并系统比较 SFT / DPO / RLVR 在字段准确率和规则通过率上的差异。`

## 为什么选投保单

- 字段空间固定，跨字段约束明确，更容易构造高信噪比 reward。
- 场景天然包含 OCR 错字、勾选框、字段缺失、条件触发等真实难点。
- 更适合体现文档智能里的 verifier engineering、reward design 和 error analysis。

## 第一版范围

第一版只做 `投保单`，不混做授信申请表、票据或其他文档任务。

- 输入
  - `pdf_page`
  - `ocr_tokens`，包含文本、bbox、页号
  - `form_type`
- 输出
  - `fields`，结构化字段结果
  - `validations`，规则校验结果

## 字段与规则重点

第一版字段集控制在 20-30 个，优先覆盖：

- 投保人/被保人基本信息
- 证件号、手机号、地址
- 险种、保额、缴费方式、缴费期限
- 受益人信息与比例
- 勾选项、签字、日期类字段

规则校验结果统一为：

- `rule_id`
- `status`
- `message`

## 默认数据路线

第一版默认使用 `公开空白模板 + 合成填充 + OCR 扰动` 生成数据，不依赖真实敏感保单数据。

数据切分按以下维度分桶：

- 模板族
- OCR 噪声等级
- 字段缺失/涂改/遮挡难例
- 规则复杂度

训练数据目前支持：

- `SFT_gold`
- `SFT_silver`
- `RL_prompt_only`
- `DPO_preference`

## 训练路线

### Phase A

- 建立 synthetic SFT baseline
- 确保输出稳定映射到统一的 `fields + validations` schema
- 输出 bucket 指标、error taxonomy 和 case export

### Phase B

- 用 verifier 对候选输出打分
- 生成 preference 对
- 训练 DPO baseline

### Phase C

- 复用同一套 verifier 组合做 composite reward
- 优先跑 `GRPO`
- 其次补 `RLOO`

## Reward 设计

第一版 reward 由以下 verifier 组成：

- `schema_reward`
- `field_match_reward`
- `normalization_reward`
- `cross_field_consistency_reward`
- `checkbox_logic_reward`
- `ocr_robustness_reward`

当前仓库已经支持 reward profile 和 ablation 入口，包括：

- `default`
- `rlvr`
- `rlvr_without_cross_field_consistency`
- `rlvr_without_checkbox_logic`

## 主指标

- Field-level F1
- Form-level exact match
- Rule pass rate
- Validation match rate
- Invalid JSON rate
- OCR-noise bucket performance

## 当前已接通能力

### 数据

- synthetic form generator，输出 `input`、`reference` 和 bucket metadata
- 支持生成 `SFT_gold` / `SFT_silver` / `RL_prompt_only`
- 支持从多候选预测生成 `DPO_preference`

### Verifier

- schema
- field match
- normalization
- cross-field consistency
- checkbox logic
- OCR robustness

### 评测

- overall 汇总指标
- bucket 维度切分统计
- error taxonomy 聚合
- failure case export
- composite reward 汇总

## 快速开始

安装开发环境：

```bash
pip install -e .[dev]
```

运行测试：

```bash
pytest
```

运行最小 smoke test：

```bash
veridoc-rl-smoke
```

也可以传入自定义 fixture 或 prediction：

```bash
veridoc-rl-smoke --fixture-path path/to/fixture.json
veridoc-rl-smoke --prediction-json '{"sample_id":"s1","fields":{},"validations":[]}'
```

## 常用脚本

生成 synthetic SFT 或 prompt-only 数据：

```bash
python scripts/generate_sft_dataset.py \
  --count 8 \
  --seed 7 \
  --task-type SFT_gold \
  --output-path outputs/sft_gold.jsonl
```

生成 preference 数据：

```bash
python scripts/generate_preference_dataset.py \
  --reference-path outputs/reference.jsonl \
  --candidate-path outputs/candidates.jsonl \
  --output-path outputs/preferences.jsonl \
  --min-margin 0.05
```

运行 phase-a 评测：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/reference.jsonl \
  --prediction-path outputs/prediction.jsonl \
  --report-path outputs/phase_a_report.json \
  --case-export-path outputs/phase_a_cases.jsonl \
  --failure-only
```

运行 reward ablation：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path outputs/reference.jsonl \
  --prediction-path outputs/prediction.jsonl \
  --report-path outputs/phase_a_report.ablation.json \
  --reward-profile rlvr_without_checkbox_logic
```

## 目录结构

```text
VeriDoc-RL/
├─ README.md
├─ pyproject.toml
├─ configs/
│  └─ experiment_matrix.yaml
├─ docs/
│  ├─ project_blueprint.md
│  ├─ resume_alignment.md
│  └─ rules_catalog.md
├─ scripts/
│  ├─ generate_sft_dataset.py
│  ├─ generate_preference_dataset.py
│  └─ run_phase_a_eval.py
├─ tests/
└─ src/
   └─ veridoc_rl/
      ├─ data/
      ├─ evaluation/
      ├─ fixtures/
      ├─ normalizers.py
      ├─ rewards/
      ├─ rules.py
      ├─ schema.py
      ├─ smoke.py
      └─ verifiers/
```

## 项目总结

- 把投保单抽取任务改写成可验证奖励问题。
- 把规则校验直接并入训练目标，而不是只做后处理。
- 复用同一套 verifier 支持 Phase A 评测、DPO preference 构造和 RLVR reward。
- 分析 OCR 噪声和模板扰动对模型性能的影响。

详细执行方案见 [docs/project_blueprint.md](docs/project_blueprint.md)。

项目定位说明见 [docs/resume_alignment.md](docs/resume_alignment.md)。

规则目录见 [docs/rules_catalog.md](docs/rules_catalog.md)。
