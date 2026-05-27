from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Kaggle kernel metadata for the target account and slug."
    )
    parser.add_argument("--kernel-dir", type=Path, required=True, help="Kernel directory path.")
    parser.add_argument("--username", required=True, help="Kaggle username to own the kernel.")
    parser.add_argument("--slug", required=True, help="Kernel slug to publish.")
    parser.add_argument("--title", required=True, help="Kernel title to publish.")
    parser.add_argument(
        "--internet",
        choices=["true", "false"],
        default="true",
        help="Whether Kaggle internet should be enabled in metadata.",
    )
    parser.add_argument(
        "--gpu",
        choices=["true", "false"],
        default="false",
        help="Whether Kaggle GPU should be enabled in metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.kernel_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = f"{args.username}/{args.slug}"
    metadata["title"] = args.title
    metadata["enable_internet"] = args.internet
    metadata["enable_gpu"] = args.gpu
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
