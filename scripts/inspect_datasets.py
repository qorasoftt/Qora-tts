from __future__ import annotations

import argparse
import os
from pathlib import Path

from qora_tts.inspection import build_report, report_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face and Kaggle dataset metadata without downloading audio."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset registry YAML file.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores the Hugging Face token.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get(args.hf_token_env)
    report = build_report(config_path=args.config, token=token)
    print(report_to_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
