# Pipeline Deep Dive

这份文档解释当前仓库主线的数据流和训练链路，不再覆盖历史分叉方案。

## 1. 任务定义

模型输入是一页投保单的 OCR token，输出是一个 JSON：

```json
{
  "sample_id": "template_a_00000",
  "fields": {
    "投保人姓名": "张三",
    "被保人出生日期": "1990-01-01"
  },
  "validations": [
    {
      "rule_id": "必填.投保人姓名",
      "status": "pass",
      "message": "投保人姓名已填写。"
    }
  ]
}
```

关键变化：

- 字段名是中文
- `rule_id` 是中文
- verifier、评测、训练语料全部围绕这套中文协议工作

## 2. 核心对象

### `input`

模型看到的输入，结构定义在 [schema.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/schema.py)。

主要字段：

- `sample_id`
- `form_type`
- `pdf_page`
- `ocr_tokens`

### `reference`

gold 答案，包含：

- `fields`
- `validations`

生成逻辑在 [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py)。

### `prediction`

模型输出经 [predictions.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/predictions.py) 清洗和解析后的结构。

它会自动：

- 去掉 fenced JSON
- 去掉 `<think>...</think>`
- 兼容英文旧键并映射成中文 canonical key

### `candidate`

DPO 构造前的多候选输出，由 [candidates.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/inference/candidates.py) 通过 OpenAI 兼容接口向 SGLang 请求。

### `preference`

DPO 用的偏好对，由 [preferences.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/preferences.py) 构造。

## 3. SFT_gold 现在怎么保证可信

这是当前仓库最重要的约束。

旧问题是：

- 内部先生成完整字段
- 再渲染 OCR
- 某些字段即使没出现在 OCR 里，gold 里也还保留

现在逻辑改成：

1. 先生成源字段
2. 再根据模板、遮挡和 hard case 决定哪些字段真正可见
3. 只基于可见字段写入 `reference.fields`
4. 再生成 `validations`
5. 导出前做一致性校验

结果：

- 被遮挡的地址、保额、申请日期不会进入 gold
- `SFT_gold` 不允许凭空编造 OCR 中不可见的自由文本字段
- 一旦 reference 和可见字段不一致，生成阶段直接报错

## 4. Verifier 与 Reward

默认 verifier 在 [form.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/verifiers/form.py)：

- `schema_reward`
- `field_match_reward`
- `normalization_reward`
- `cross_field_consistency_reward`
- `checkbox_logic_reward`
- `ocr_robustness_reward`

它们共同依赖 [form_spec.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/form_spec.py) 和 [normalizers.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/normalizers.py)。

同一套 verifier 被复用在：

- DPO 候选排序
- RL reward
- 最终评测

## 5. 三阶段训练链路

### Phase A: SFT

- 输入：`SFT_gold`
- 目标：学会中文字段输出格式、规则输出格式、基础抽取行为
- 产物：`phase_a_sft/checkpoints`

### Phase B: DPO

当前默认已不是 baseline 采样。

现在是：

1. 用 `phase_a_sft` checkpoint 对应的 adapter 挂到 SGLang
2. 从这个 SFT 模型采样多候选
3. 用 verifier score 排序
4. 生成 `chosen/rejected`
5. 训练 DPO

这样做的原因是：

- baseline 多回答之间差异太小
- preference 容易为空
- SFT 之后再采样，候选质量和分差更稳定

### Phase C: RL

- 输入：`RL_prompt_only`
- reward：同一套 verifier 组合分数
- 当前默认算法：`grpo`

## 6. 为什么关闭 Qwen3 thinking

Qwen3 默认可能在输出前先给出 `<think>...</think>`。

这会导致：

- 解析更难
- verifier 前面拿不到合法 JSON
- reward 容易趋同

当前仓库用了双保险：

- 请求侧默认 `enable_thinking=false`
- 解析侧继续清洗 `<think>...</think>`

## 7. Pipeline 当前默认行为

主配置在 [pipeline.autodl.qwen3_1p7.yaml](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/configs/pipeline.autodl.qwen3_1p7.yaml)。

关键默认值：

- `preference_source: sft_adapter`
- `disable_thinking: true`
- `preference_disable_thinking: true`
- `enable_post_train_eval: false`

所以一键 pipeline 现在负责：

- baseline 候选/评测
- SFT 训练
- DPO 候选构造与训练
- RL 训练

但默认不负责：

- phase A/B/C checkpoint 的训练后评测

## 8. 文件映射

最值得关注的文件：

- [form_spec.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/form_spec.py)
- [synthetic.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/data/synthetic.py)
- [prompting.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/training/prompting.py)
- [predictions.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/predictions.py)
- [form.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/verifiers/form.py)
- [stages.py](/home/alvis/projects/llm-study/VeriDoc-RL/VeriDoc-RL/src/veridoc_rl/orchestration/stages.py)

## 9. 调试顺序建议

如果链路出问题，按这个顺序看：

1. `SFT_gold` 是否合理
2. `predictions.py` 是否成功解析输出
3. `phase_b_dpo/candidates.jsonl` 是否真的来自 SFT adapter
4. reward margin 是否足够
5. SGLang 是否真的关闭了 thinking 并挂载了 LoRA
