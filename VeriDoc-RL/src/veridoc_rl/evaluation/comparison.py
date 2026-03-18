from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_COMPARISON_METRICS: tuple[str, ...] = (
    "field_f1",
    "form_exact_match",
    "rule_pass_rate",
    "validation_match_rate",
    "total_reward",
)
_DEFAULT_COLORS: tuple[str, ...] = ("#22577a", "#38a3a5", "#57cc99", "#80ed99", "#c7f9cc")


@dataclass(slots=True, frozen=True)
class ReportSnapshot:
    label: str
    path: Path
    payload: dict[str, Any]


def load_report_snapshot(label: str, path: Path) -> ReportSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report at {path} must be a JSON object.")
    return ReportSnapshot(label=label, path=path, payload=payload)


def compare_report_snapshots(
    snapshots: list[ReportSnapshot],
    *,
    bucket_dimension: str = "ocr_noise_level",
    bucket_metric: str = "field_f1",
) -> dict[str, Any]:
    overall_table = [
        {
            "label": snapshot.label,
            **_extract_overall_metrics(snapshot.payload.get("overall")),
        }
        for snapshot in snapshots
    ]
    return {
        "reports": [
            {"label": snapshot.label, "path": str(snapshot.path)}
            for snapshot in snapshots
        ],
        "overall_table": overall_table,
        "best_by_metric": _compute_metric_leaders(overall_table),
        "bucket_dimension": bucket_dimension,
        "bucket_metric": bucket_metric,
        "bucket_comparison": _compare_bucket_metric(
            snapshots,
            dimension=bucket_dimension,
            metric=bucket_metric,
        ),
        "failure_case_digest": _collect_failure_case_digest(snapshots),
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    metric_leaders = comparison.get("best_by_metric", {})
    overall_rows = comparison.get("overall_table", [])
    bucket_dimension = str(comparison.get("bucket_dimension", "ocr_noise_level"))
    bucket_metric = str(comparison.get("bucket_metric", "field_f1"))

    lines = [
        "# Phase Report Comparison",
        "",
        "## Overall Metrics",
    ]
    for row in overall_rows:
        lines.append(
            (
                f"- `{row['label']}`: field_f1={row.get('field_f1', 0.0):.3f}, "
                f"form_exact_match={row.get('form_exact_match', 0.0):.3f}, "
                f"rule_pass_rate={row.get('rule_pass_rate', 0.0):.3f}, "
                f"total_reward={row.get('total_reward', 0.0):.3f}"
            )
        )

    lines.extend(["", "## Metric Leaders"])
    for metric, leader in metric_leaders.items():
        lines.append(f"- `{metric}`: `{leader['label']}` ({leader['value']:.3f})")

    lines.extend(
        [
            "",
            f"## Bucket Comparison: `{bucket_dimension}` / `{bucket_metric}`",
        ]
    )
    for bucket_name, values in comparison.get("bucket_comparison", {}).items():
        metrics_text = ", ".join(
            f"{label}={float(metric_value):.3f}" for label, metric_value in values.items()
        )
        lines.append(f"- `{bucket_name}`: {metrics_text}")

    lines.extend(["", "## Failure Case Digest"])
    digest = comparison.get("failure_case_digest", {})
    if not digest:
        lines.append("- No failure cases were exported in the compared reports.")
    else:
        for label, items in digest.items():
            if not items:
                lines.append(f"- `{label}`: no failure cases.")
                continue
            first = items[0]
            lines.append(
                (
                    f"- `{label}`: top failure `{first['sample_id']}` with taxonomy "
                    f"{', '.join(first['taxonomy']) or 'none'}"
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_comparison_artifacts(
    output_dir: Path,
    comparison: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    rule_chart_path = output_dir / "rule_pass_rate_comparison.svg"
    bucket_chart_path = output_dir / f"{comparison['bucket_dimension']}_{comparison['bucket_metric']}.svg"

    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    rule_chart_path.write_text(
        _render_overall_metric_chart(
            comparison.get("overall_table", []),
            metric="rule_pass_rate",
            title="Rule Pass Rate Comparison",
        ),
        encoding="utf-8",
    )
    bucket_chart_path.write_text(
        _render_bucket_chart(
            comparison.get("bucket_comparison", {}),
            title=f"{comparison['bucket_dimension']} / {comparison['bucket_metric']}",
        ),
        encoding="utf-8",
    )
    return {
        "comparison_json": json_path,
        "comparison_markdown": markdown_path,
        "rule_pass_rate_chart": rule_chart_path,
        "bucket_chart": bucket_chart_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple phase report JSON files and export summary artifacts.")
    parser.add_argument(
        "--report",
        dest="reports",
        action="append",
        required=True,
        help="Report spec in the form label=path/to/report.json. Repeat for each experiment.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison JSON/markdown/SVG artifacts.")
    parser.add_argument("--bucket-dimension", default="ocr_noise_level", help="Bucket dimension to compare in the grouped chart.")
    parser.add_argument("--bucket-metric", default="field_f1", help="Bucket metric to compare in the grouped chart.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    snapshots = [
        load_report_snapshot(label=label, path=path)
        for label, path in (_parse_report_spec(spec) for spec in args.reports)
    ]
    comparison = compare_report_snapshots(
        snapshots,
        bucket_dimension=args.bucket_dimension,
        bucket_metric=args.bucket_metric,
    )
    write_comparison_artifacts(args.output_dir, comparison)
    return 0


def _parse_report_spec(spec: str) -> tuple[str, Path]:
    label, separator, path_text = spec.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise ValueError("Each --report value must be formatted as label=path.")
    return label.strip(), Path(path_text.strip())


def _extract_overall_metrics(overall: Any) -> dict[str, float | int]:
    payload = overall if isinstance(overall, dict) else {}
    return {
        "sample_count": int(payload.get("sample_count", 0)),
        "failure_count": int(payload.get("failure_count", 0)),
        **{
            metric: float(payload.get(metric, 0.0))
            for metric in DEFAULT_COMPARISON_METRICS
        },
    }


def _compute_metric_leaders(overall_table: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    leaders: dict[str, dict[str, Any]] = {}
    for metric in DEFAULT_COMPARISON_METRICS:
        if not overall_table:
            continue
        best_row = max(overall_table, key=lambda row: float(row.get(metric, 0.0)))
        leaders[metric] = {
            "label": best_row["label"],
            "value": float(best_row.get(metric, 0.0)),
        }
    return leaders


def _compare_bucket_metric(
    snapshots: list[ReportSnapshot],
    *,
    dimension: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for snapshot in snapshots:
        bucket_metrics = snapshot.payload.get("bucket_metrics", {})
        if not isinstance(bucket_metrics, dict):
            continue
        dimension_payload = bucket_metrics.get(dimension, {})
        if not isinstance(dimension_payload, dict):
            continue
        for bucket_name, metrics in dimension_payload.items():
            if not isinstance(metrics, dict):
                continue
            comparison.setdefault(str(bucket_name), {})[snapshot.label] = float(metrics.get(metric, 0.0))
    return dict(sorted(comparison.items()))


def _collect_failure_case_digest(snapshots: list[ReportSnapshot]) -> dict[str, list[dict[str, Any]]]:
    digest: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        failure_cases = snapshot.payload.get("failure_cases", [])
        if not isinstance(failure_cases, list):
            digest[snapshot.label] = []
            continue
        digest[snapshot.label] = [
            {
                "sample_id": str(item.get("sample_id", "")),
                "taxonomy": [str(tag) for tag in item.get("taxonomy", []) if isinstance(tag, str)],
            }
            for item in failure_cases[:3]
            if isinstance(item, dict)
        ]
    return digest


def _render_overall_metric_chart(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
) -> str:
    width = 720
    height = 420
    left = 80
    bottom = 60
    top = 40
    chart_width = width - left - 40
    chart_height = height - top - bottom
    bar_width = chart_width / max(len(rows), 1) * 0.6
    gap = chart_width / max(len(rows), 1) * 0.4

    labels = []
    bars = []
    for index, row in enumerate(rows):
        value = max(0.0, min(1.0, float(row.get(metric, 0.0))))
        x = left + index * (bar_width + gap) + gap / 2
        bar_height = chart_height * value
        y = top + chart_height - bar_height
        color = _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]
        labels.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="12">{row["label"]}</text>'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12">{value:.2f}</text>'
        )
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="4" />'
        )

    return _wrap_svg(
        width=width,
        height=height,
        title=title,
        body=[
            _svg_axes(width=width, height=height, left=left, top=top, chart_height=chart_height, bottom=bottom),
            *bars,
            *labels,
        ],
    )


def _render_bucket_chart(
    bucket_values: dict[str, dict[str, float]],
    *,
    title: str,
) -> str:
    width = 860
    height = 460
    left = 90
    bottom = 70
    top = 40
    chart_width = width - left - 40
    chart_height = height - top - bottom
    bucket_names = list(bucket_values)
    labels = sorted({label for values in bucket_values.values() for label in values})
    group_width = chart_width / max(len(bucket_names), 1)
    bar_width = (group_width * 0.8) / max(len(labels), 1)

    bars: list[str] = []
    annotations: list[str] = []
    for bucket_index, bucket_name in enumerate(bucket_names):
        base_x = left + bucket_index * group_width + group_width * 0.1
        annotations.append(
            f'<text x="{base_x + group_width * 0.4:.1f}" y="{height - 22}" text-anchor="middle" font-size="12">{bucket_name}</text>'
        )
        for label_index, label in enumerate(labels):
            value = max(0.0, min(1.0, float(bucket_values[bucket_name].get(label, 0.0))))
            x = base_x + label_index * bar_width
            bar_height = chart_height * value
            y = top + chart_height - bar_height
            color = _DEFAULT_COLORS[label_index % len(_DEFAULT_COLORS)]
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3" />'
            )
            annotations.append(
                f'<text x="{x + (bar_width - 4) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="10">{value:.2f}</text>'
            )

    legend = [
        f'<rect x="{left + index * 150:.1f}" y="{height - 48}" width="14" height="14" fill="{_DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]}" rx="2" />'
        f'<text x="{left + index * 150 + 22:.1f}" y="{height - 36}" font-size="12">{label}</text>'
        for index, label in enumerate(labels)
    ]

    return _wrap_svg(
        width=width,
        height=height,
        title=title,
        body=[
            _svg_axes(width=width, height=height, left=left, top=top, chart_height=chart_height, bottom=bottom),
            *bars,
            *annotations,
            *legend,
        ],
    )


def _svg_axes(
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    chart_height: int,
    bottom: int,
) -> str:
    axis_y = top + chart_height
    return "\n".join(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="#334155" stroke-width="1.5" />',
            f'<line x1="{left}" y1="{axis_y}" x2="{width - 30}" y2="{axis_y}" stroke="#334155" stroke-width="1.5" />',
            f'<text x="{left - 24}" y="{top + 6}" font-size="12">1.0</text>',
            f'<text x="{left - 24}" y="{axis_y + 4}" font-size="12">0.0</text>',
            f'<text x="{left}" y="{height - bottom + 48}" font-size="12" fill="#475569">Generated by VeriDoc-RL</text>',
        ]
    )


def _wrap_svg(*, width: int, height: int, title: str, body: list[str]) -> str:
    content = "\n".join(body)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc" />',
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">{title}</text>',
            content,
            "</svg>",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
