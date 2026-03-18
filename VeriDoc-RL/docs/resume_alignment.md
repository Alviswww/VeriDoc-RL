# VeriDoc-RL 项目定位说明

## 1. 项目定位

VeriDoc-RL 更适合被描述为一个 `verifier-guided post-training` 项目，而不只是普通的文档抽取工程。

它覆盖的核心主题包括：

- 大模型后训练
- SFT / DPO / RLVR
- 文档智能算法
- verifier / reward / evaluation design

这个项目的价值不在“再做一个表单抽取任务”，而在于把真实业务规则转成可验证、可训练、可评测的后训练信号。

## 2. 为什么选择投保单场景

这个场景更容易体现以下能力：

- 抽取和校验联动，而不是只做信息提取
- verifier 可程序化实现，技术论证更硬
- OCR 噪声、勾选逻辑、条件字段是实际难点
- reward 设计和 error analysis 更容易讲清楚

## 3. 可复用的方法论

这个项目沉淀的不是某一个垂类模板本身，而是一套可以迁移的方法论：

1. 把制式文档任务定义成“结构化生成 + 规则校验”的统一输出问题。
2. 把字段约束、互斥逻辑和跨字段一致性显式化为 verifier。
3. 复用同一套 verifier 支持 Phase A 评测、DPO preference 构造和 RLVR reward。
4. 围绕 bucket、error taxonomy 和 case analysis 形成实验闭环。

## 4. 对外介绍时可强调的版本

### 4.1 后训练视角

- 设计并实现 `VeriDoc-RL`，面向投保单 OCR 结果构建统一的 `fields + validations` 输出 schema，将字段完整性、标准化、跨字段一致性和勾选逻辑转化为 verifier-guided reward，系统比较 SFT、DPO 与 RLVR 在字段准确率和规则通过率上的差异。
- 构建基于模板族、OCR 噪声等级和规则复杂度的评测集，围绕字段漏抽、标准化错误、关系字段错误、checkbox 冲突等问题开展 reward ablation 和 case analysis，形成可复现的表单后训练评测闭环。

### 4.2 文档智能视角

- 将制式表单抽取任务建模为“结构化生成 + 规则校验”的联合优化问题，复用同一套 verifier 支持偏好数据构造、RLVR reward 与最终评测，提升 OCR 噪声和模板变体下的抽取鲁棒性。

## 5. 90 秒项目讲法

VeriDoc-RL 这个项目把题收敛到投保单这种制式文件，因为它的字段空间固定、跨字段规则明确，而且天然有 OCR 错字、勾选框、条件字段这些真实难点。基于这个判断，项目把任务定义成统一的 `fields + validations` 输出，让模型不仅要抽对字段，还要给出规则校验结果。随后把字段完整性、标准化、跨字段一致性、checkbox 逻辑这些约束实现成 verifier，并让同一套 verifier 同时服务于 DPO 数据构造、RLVR reward 和最终评测。这个项目的重点不是单纯跑一个 RL 框架，而是把真实业务约束系统化地改写成后训练问题。

## 6. 对外表述时需要克制的内容

以下结论必须建立在实验结果之上，再公开写出：

- “RLVR 显著优于 DPO”
- “OCR robustness reward 显著提升复杂样本表现”
- “cross-field consistency verifier 带来 x% 的规则通过率提升”
- “表单级 exact match 提升 x%”
