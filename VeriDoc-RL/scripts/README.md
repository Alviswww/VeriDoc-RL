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
- `python scripts/generate_experiment_plan.py`
  - 读取 `configs/experiment_matrix.yaml`
  - 展开 blueprint 中的实验队列
  - 输出 JSON/Markdown 两种实验计划产物
- `python scripts/compare_phase_reports.py`
  - 读取多份 phase report JSON
  - 输出 overall 对比、bucket 对比、failure digest
  - 生成规则通过率对比图和 OCR-noise bucket 图
- `python scripts/prepare_training_data.py`
  - 把 `SFT_gold` / `DPO_preference` / `RL_prompt_only` 数据转成统一训练语料
  - 输出 Phase A SFT、Phase B DPO、Phase C RLVR 可直接消费的 JSONL
- `python scripts/generate_training_manifests.py`
  - 读取 `experiment_matrix.yaml` 里的训练 runtime 配置
  - 生成 `phase_b_dpo` / `phase_c_grpo` / `phase_c_rloo` 的 `verl` 风格 manifest bundle
- `python scripts/prepare_verl_runtime.py`
  - 读取单个 `manifest.json`
  - 生成 `launch.sh` 与 `runtime_plan.json`
  - 对 `phase_c_grpo` / `phase_c_rloo` 可桥接到 `verl.trainer.main_ppo`
