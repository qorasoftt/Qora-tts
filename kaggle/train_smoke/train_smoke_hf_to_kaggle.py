from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
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


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict) and "path" in value:
            normalized[key] = {k: value[k] for k in value if k != "bytes"}
        else:
            normalized[key] = value
    return normalized


def sample_rows(
    repo_id: str,
    config_name: str | None,
    token: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        path=repo_id,
        name=config_name,
        split="train",
        streaming=True,
        token=token,
    )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if index >= max_rows:
            break
        rows.append(normalize_row(row))
    return rows


def inspect_dataset(
    repo_id: str,
    token: str | None,
    max_rows: int,
    preferred_configs: Iterable[str] | None = None,
) -> dict[str, Any]:
    api = HfApi(token=token) if token else HfApi()
    info = api.dataset_info(repo_id=repo_id, token=token)
    payload: dict[str, Any] = {
        "repo_id": repo_id,
        "gated": info.gated,
        "private": info.private,
        "downloads": info.downloads,
        "hf_token_present": token is not None,
    }

    configs = get_dataset_config_names(path=repo_id, token=token)
    payload["configs"] = configs

    chosen_config: str | None = None
    if configs:
        config_candidates = list(preferred_configs or [])
        for candidate in config_candidates:
            if candidate in configs:
                chosen_config = candidate
                break
        if chosen_config is None:
            chosen_config = configs[0]

    builder = load_dataset_builder(path=repo_id, name=chosen_config, token=token)
    payload["builder_name"] = builder.builder_name
    payload["config_name"] = chosen_config
    payload["splits"] = sorted(builder.info.splits.keys()) if builder.info.splits else []
    payload["features"] = list(builder.info.features.keys()) if builder.info.features else []
    payload["sample_rows"] = sample_rows(
        repo_id=repo_id,
        config_name=chosen_config,
        token=token,
        max_rows=max_rows,
    )
    return payload


def main() -> None:
    token = read_secret("HF_TOKEN")
    max_rows = int(os.environ.get("SMOKE_MAX_ROWS", "4"))

    report = {
        "mbaza": None,
        "waxal_candidates": [],
    }

    try:
        report["mbaza"] = inspect_dataset(
            repo_id="mbazaNLP/kinyarwanda-tts-dataset",
            token=token,
            max_rows=max_rows,
        )
    except Exception as exc:
        report["mbaza"] = {"error": f"{type(exc).__name__}: {exc}"}

    waxal_preferred = ["lug_tts", "kik_tts", "nyn_tts", "swa_tts"]
    try:
        report["waxal_candidates"].append(
            inspect_dataset(
                repo_id="google/WaxalNLP",
                token=token,
                max_rows=max_rows,
                preferred_configs=waxal_preferred,
            )
        )
    except Exception as exc:
        report["waxal_candidates"].append({"error": f"{type(exc).__name__}: {exc}"})

    output_path = "/kaggle/working/qora_tts_smoke_report.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print(json.dumps(report, indent=2, ensure_ascii=True))
    print({"output_path": output_path})


if __name__ == "__main__":
    main()
