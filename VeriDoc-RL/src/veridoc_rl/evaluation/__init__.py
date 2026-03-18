from __future__ import annotations

from veridoc_rl.evaluation.metrics import (
    canonicalize_field_value,
    compute_field_level_metrics,
    compute_form_exact_match,
    compute_invalid_json_rate,
    compute_rule_pass_rate,
    compute_validation_match_rate,
    evaluate_prediction,
)
from veridoc_rl.evaluation.reporting import (
    BUCKET_DIMENSIONS,
    ERROR_TAXONOMY,
    CaseAnalysis,
    DatasetEvaluationReport,
    build_evaluation_entries,
    evaluate_case,
    evaluate_dataset,
    evaluate_dataset_files,
    export_case_records,
    load_jsonl,
    write_report,
)
from veridoc_rl.evaluation.comparison import (
    ReportSnapshot,
    compare_report_snapshots,
    load_report_snapshot,
    render_comparison_markdown,
    write_comparison_artifacts,
)

__all__ = [
    "BUCKET_DIMENSIONS",
    "ERROR_TAXONOMY",
    "CaseAnalysis",
    "DatasetEvaluationReport",
    "ReportSnapshot",
    "build_evaluation_entries",
    "canonicalize_field_value",
    "compare_report_snapshots",
    "compute_field_level_metrics",
    "compute_form_exact_match",
    "compute_invalid_json_rate",
    "compute_rule_pass_rate",
    "compute_validation_match_rate",
    "evaluate_case",
    "evaluate_dataset",
    "evaluate_dataset_files",
    "evaluate_prediction",
    "export_case_records",
    "load_jsonl",
    "load_report_snapshot",
    "render_comparison_markdown",
    "write_comparison_artifacts",
    "write_report",
]
