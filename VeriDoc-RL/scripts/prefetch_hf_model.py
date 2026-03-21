from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from veridoc_rl.model_defaults import DEFAULT_BASELINE_MODEL

DEFAULT_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.txt",
    "tokenizer*",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into a stable local directory."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_BASELINE_MODEL,
        help="Hugging Face repo id to download.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Local directory that will contain the snapshot.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Optional allow pattern. Repeat to override the defaults.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.target_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(args.target_dir),
        allow_patterns=list(args.allow_patterns or DEFAULT_ALLOW_PATTERNS),
    )
    print(f"Model snapshot ready at {args.target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
