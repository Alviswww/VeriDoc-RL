# evaluation

Phase A 评测现在已经接通三类产出：

- 核心指标：`Field-level F1`、`Form exact match`、`Rule pass rate`、`Invalid JSON rate`
- 分桶统计：模板族、OCR 噪声等级、难例类型、规则复杂度
- 分析导出：error taxonomy 聚合、case export JSONL 和 composite reward

推荐入口：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path data/reference.jsonl \
  --prediction-path data/prediction.jsonl \
  --report-path outputs/phase_a_report.json \
  --case-export-path outputs/phase_a_cases.jsonl \
  --failure-only
```

如果要做 Week 7 的 reward ablation，可以切换 reward profile：

```bash
python scripts/run_phase_a_eval.py \
  --reference-path data/reference.jsonl \
  --prediction-path data/prediction.jsonl \
  --report-path outputs/phase_a_report.ablation.json \
  --reward-profile rlvr_without_checkbox_logic
```
