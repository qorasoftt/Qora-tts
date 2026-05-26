from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROBE_SCRIPT = """\
from __future__ import annotations

import os

from datasets import get_dataset_config_names
from huggingface_hub import HfApi

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None


def read_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if UserSecretsClient is not None:
        try:
            value = UserSecretsClient().get_secret(name)
        except Exception:
            value = None
        if value:
            return value
    return None


def main() -> None:
    token = read_secret("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    targets = [
        "mbazaNLP/kinyarwanda-tts-dataset",
        "google/WaxalNLP",
    ]

    for repo_id in targets:
        try:
            info = api.dataset_info(repo_id=repo_id, token=token)
            payload = {
                "repo_id": repo_id,
                "gated": info.gated,
                "private": info.private,
                "downloads": info.downloads,
                "hf_token_present": token is not None,
            }
            print(payload)
            configs = get_dataset_config_names(path=repo_id, token=token)
            print({"repo_id": repo_id, "configs": configs[:20], "config_count": len(configs)})
        except Exception as exc:
            print({"repo_id": repo_id, "config_error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Kaggle probe scaffold for HF dataset access."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/probe"),
        help="Directory where kernel metadata and the probe script are written.",
    )
    return parser.parse_args()


def resolve_kaggle_username() -> str:
    completed = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", maxsplit=1)[1].strip()
    raise RuntimeError("Unable to resolve Kaggle username from `kaggle config view`.")


def main() -> int:
    args = parse_args()
    username = resolve_kaggle_username()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": f"{username}/qora-tts-dataset-probe",
        "title": "Qora TTS Dataset Probe",
        "code_file": "probe_hf_to_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "probe_hf_to_kaggle.py").write_text(PROBE_SCRIPT, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
