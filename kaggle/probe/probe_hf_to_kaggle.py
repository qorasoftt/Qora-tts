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
