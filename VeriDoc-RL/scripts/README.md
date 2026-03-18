# scripts

当前已接通的脚本：

- `python scripts/generate_sft_dataset.py`
  - 生成带 bucket 元数据的 synthetic 数据
  - 支持 `SFT_gold`、`SFT_silver`、`RL_prompt_only`
- `python scripts/generate_preference_dataset.py`
  - 读取 `reference` 和多候选 `prediction` JSONL
  - 用同一套 verifier + composite reward 生成 `DPO_preference`
- `python scripts/run_phase_a_eval.py`
  - 读取 reference/prediction JSONL
  - 跑 verifier suite
  - 输出 bucket 评测、error taxonomy、case export 和 composite reward
