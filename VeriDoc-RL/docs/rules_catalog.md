# 投保单规则目录

下面这组规则用于统一：

- 数据标注真值
- verifier 实现
- DPO 候选打分
- RLVR reward 组合
- 最终评测口径

## 规则列表

1. `required.policyholder_name`
   - 投保人姓名必填
2. `required.policyholder_id_number`
   - 投保人证件号必填
3. `required.insured_name`
   - 被保人姓名必填
4. `format.policyholder_phone`
   - 投保人手机号必须标准化为 11 位数字
5. `format.policyholder_id_number`
   - 证件号格式必须合法
6. `format.application_date`
   - 申请日期必须标准化为 `YYYY-MM-DD`
7. `consistency.birth_date_vs_id_number`
   - 出生日期与证件号中的出生日期片段一致
8. `consistency.beneficiary_ratio_sum`
   - 受益人比例之和必须为 100
9. `consistency.policyholder_insured_relation`
   - 投保人与被保人关系字段必须落在允许集合中
10. `consistency.product_payment_combo`
   - 险种与缴费方式/缴费期限组合必须合理
11. `checkbox.payment_mode_exclusive`
   - 缴费方式互斥勾选项不能同时成立
12. `checkbox.auto_debit_requires_account`
   - 若勾选自动扣款，则必须提供扣款账户信息

## 状态定义

- `pass`
- `fail`
- `not_applicable`

## 设计要求

- 每条规则必须有唯一 `rule_id`
- 每条规则必须可程序化判断
- 每条规则必须可映射到 verifier
- 训练、偏好数据构造和评测必须复用同一规则定义
