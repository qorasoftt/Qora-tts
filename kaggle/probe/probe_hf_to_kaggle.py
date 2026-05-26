from __future__ import annotations

import os

from datasets import get_dataset_config_names
from huggingface_hub import HfApi


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    token = require_env("HF_TOKEN")
    api = HfApi(token=token)
    targets = [
        "mbazaNLP/kinyarwanda-tts-dataset",
        "google/WaxalNLP",
    ]

    for repo_id in targets:
        info = api.dataset_info(repo_id=repo_id, token=token)
        print(
            {
                "repo_id": repo_id,
                "gated": info.gated,
                "private": info.private,
                "downloads": info.downloads,
            }
        )
        try:
            configs = get_dataset_config_names(path=repo_id, token=token)
            print({"repo_id": repo_id, "configs": configs[:20], "config_count": len(configs)})
        except Exception as exc:
            print({"repo_id": repo_id, "config_error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
