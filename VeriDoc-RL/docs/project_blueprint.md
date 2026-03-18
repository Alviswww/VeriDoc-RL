# VeriDoc-RL 项目执行蓝图

## 1. 项目目标

把“投保单制式文件抽取”从普通的 SFT 任务升级成一个带可验证奖励的后训练项目。项目最终回答 4 个问题：

1. SFT baseline 在投保单抽取和规则校验上主要失败在哪里？
2. 哪些业务规则能被稳定转成 verifier？
3. verifier 组成的 reward 能否提升字段准确率和规则通过率？
4. RLVR 相比 SFT 和 DPO，在哪些模板和 OCR 噪声场景下更有增益？

## 2. 为什么这个题值得做

这个方向和文档解析、结构化抽取、多模态数据构建与评测等问题高度同源，并且更聚焦 verifier 价值：

- 字段空间固定
- 业务规则强
- OCR 噪声更真实
- case 分析和 reward 解释更直接

## 3. 范围控制

### 3.1 第一版只做投保单

第一版不做：

- 其他金融申请表
- 票据/KV 混合任务
- 原始图像到端到端多模态抽取
- 页级溯源或定位信息建模

### 3.2 第一版目标固定为抽取+校验

模型输出必须同时包含两部分：

- `fields`
- `validations`

这样 verifier 不只是后处理，而是训练目标的一部分。

## 4. 输入输出定义

### 4.1 输入

每条样本统一使用 OCR 后的结构化结果：

```json
{
  "sample_id": "tpl_a_0001",
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

### 4.2 输出

```json
{
  "sample_id": "tpl_a_0001",
  "fields": {
    "policyholder_name": "张三",
    "policyholder_id_number": "440101199001011234",
    "insured_name": "李四",
    "insured_birth_date": "1992-08-10",
    "sum_assured": "300000",
    "premium_payment_mode": "annual"
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

### 4.3 字段范围

第一版字段集控制在 20-30 个，建议至少包含：

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

## 5. 数据构建方案

### 5.1 数据来源

默认路线：

- 公开空白投保单模板
- 自定义字段填充器
- OCR 扰动注入器

### 5.2 样本类型

训练数据分成三类：

1. `SFT_gold`
   - 高质量字段和规则真值
2. `SFT_silver`
   - 模板批量合成数据
3. `RL_prompt_only`
   - 用于 rollout 和 verifier 打分

### 5.3 分桶方式

不再使用长度分桶。统一按以下维度评测：

- 模板族
- OCR 噪声等级
- 难例类型
- 规则复杂度

### 5.4 Error taxonomy

建议至少维护下面 8 类错误：

1. 漏抽字段
2. 值错误
3. 标准化错误
4. 关系字段错误
5. 勾选逻辑错误
6. 规则误判
7. OCR 噪声脆弱
8. JSON/schema 非法

## 6. 训练路线

### 6.1 Phase A: SFT baseline

目标：

- 先把统一 schema 跑通
- 形成首版 error taxonomy

产出：

- Field-level F1
- Form-level exact match
- Invalid JSON rate
- 失败 case 集

### 6.2 Phase B: DPO

流程：

- 对同一输入采样多个 candidate
- 用 verifier 组合打分
- 构造 chosen/rejected
- 训练 DPO

### 6.3 Phase C: RLVR

流程：

- 复用 DPO 阶段同一套 verifier
- 组合成 composite reward
- 优先做 `GRPO`
- 其次做 `RLOO`

## 7. Verifier / reward 设计

### 7.1 Reward 组件

1. `schema_reward`
   - JSON 可解析
   - 字段类型正确
   - 必填字段完整
2. `field_match_reward`
   - 字段 exact/soft match
3. `normalization_reward`
   - 日期、手机号、证件号、金额标准化正确
4. `cross_field_consistency_reward`
   - 年龄/生日一致
   - 投保人与被保人关系一致
   - 受益人比例和正确
   - 险种与缴费字段组合合理
5. `checkbox_logic_reward`
   - 互斥项不冲突
   - 条件触发字段完整
6. `ocr_robustness_reward`
   - 对错字、断词、框偏移后的输出稳定性

### 7.2 初始权重建议

- schema: `0.20`
- field match: `0.30`
- normalization: `0.15`
- cross field consistency: `0.20`
- checkbox logic: `0.10`
- ocr robustness: `0.05`

### 7.3 必做 ablation

1. SFT
2. SFT + DPO
3. SFT + RLVR
4. RLVR 去掉 `cross_field_consistency_reward`
5. RLVR 去掉 `checkbox_logic_reward`

## 8. 评测设计

### 8.1 主指标

- Field-level F1
- Form-level exact match
- Rule pass rate
- Invalid JSON rate
- OCR-noise bucket performance

### 8.2 分桶分析

必须按以下维度切分：

- 模板族
- OCR 噪声等级
- 规则复杂度
- 难例类型

### 8.3 展示方式

至少输出：

- 一张总表
- 一张 OCR-noise bucket 图
- 一张规则通过率对比图
- 三个失败 case 分析

## 9. 8 周执行节奏

### Week 1

- 固化字段 schema
- 整理规则目录
- 设计模板与合成数据格式

### Week 2

- 构建第一批 SFT 数据
- 打通 SFT baseline

### Week 3

- 固化 error taxonomy
- 补 OCR 扰动难例

### Week 4

- 实现 schema/field/normalization verifier

### Week 5

- 实现 cross-field / checkbox / OCR robustness verifier
- 生成 preference 数据

### Week 6

- 跑 DPO baseline
- 跑首轮 RLVR

### Week 7

- 做 reward ablation
- 做模板/噪声/规则复杂度切分评测

### Week 8

- 整理图表
- 固化项目总结和对外表述

## 10. 对外表述

可以把项目概括成：

我把投保单抽取任务建模成“结构化抽取 + 规则校验”的统一输出问题。制式表单的字段和规则稳定性更高，所以我把业务规则直接实现成 verifier，再用同一套 verifier 同时服务于 DPO 数据构造、RLVR reward 和最终评测。这样项目的重点不是跑通某个 RL 框架，而是把真实业务约束转成可训练、可解释、可评测的后训练系统。
