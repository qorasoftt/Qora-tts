from __future__ import annotations

import json
import subprocess
from csv import DictReader
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    hf_repo_id: str
    kaggle_search: list[str]
    notes: str


@dataclass(frozen=True)
class HuggingFaceDatasetInspection:
    repo_id: str
    private: bool | None
    gated: str | bool | None
    disabled: bool | None
    downloads: int | None
    likes: int | None
    configs: list[str] | None
    config_error: str | None


@dataclass(frozen=True)
class KaggleSearchInspection:
    query: str
    rows: list[dict[str, str]]


def load_dataset_specs(config_path: Path) -> list[DatasetSpec]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "datasets" not in raw:
        raise ValueError(f"Expected top-level 'datasets' list in {config_path}")

    specs = raw["datasets"]
    if not isinstance(specs, list):
        raise ValueError(f"Expected 'datasets' to be a list in {config_path}")

    return [
        DatasetSpec(
            name=str(item["name"]),
            role=str(item["role"]),
            hf_repo_id=str(item["hf_repo_id"]),
            kaggle_search=[str(entry) for entry in item.get("kaggle_search", [])],
            notes=str(item["notes"]),
        )
        for item in specs
    ]


def inspect_hugging_face_dataset(
    api: HfApi,
    repo_id: str,
    token: str | None,
) -> HuggingFaceDatasetInspection:
    info = api.dataset_info(repo_id=repo_id, token=token)
    config_names: list[str] | None = None
    config_error: str | None = None
    try:
        config_names = sorted(get_dataset_config_names(path=repo_id, token=token))
    except Exception as exc:  # datasets raises inconsistent exception types here
        config_error = f"{type(exc).__name__}: {exc}"

    return HuggingFaceDatasetInspection(
        repo_id=repo_id,
        private=info.private,
        gated=info.gated,
        disabled=info.disabled,
        downloads=info.downloads,
        likes=info.likes,
        configs=config_names,
        config_error=config_error,
    )


def inspect_kaggle_search(query: str) -> KaggleSearchInspection:
    command = ["kaggle", "datasets", "list", "-s", query, "-v"]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = parse_kaggle_csv(completed.stdout)
    return KaggleSearchInspection(query=query, rows=rows)


def parse_kaggle_csv(raw: str) -> list[dict[str, str]]:
    stripped = raw.strip()
    if not stripped:
        return []

    return [dict(row) for row in DictReader(stripped.splitlines())]


def build_report(config_path: Path, token: str | None = None) -> dict[str, Any]:
    api = HfApi()
    specs = load_dataset_specs(config_path)
    report: list[dict[str, Any]] = []

    for spec in specs:
        try:
            hf_inspection = inspect_hugging_face_dataset(
                api=api,
                repo_id=spec.hf_repo_id,
                token=token,
            )
            hf_payload: dict[str, Any] = asdict(hf_inspection)
        except HfHubHTTPError as exc:
            hf_payload = {
                "repo_id": spec.hf_repo_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

        kaggle_payload = [
            {
                "query": inspection.query,
                "rows": inspection.rows[:5],
            }
            for inspection in (inspect_kaggle_search(query) for query in spec.kaggle_search)
        ]

        report.append(
            {
                "spec": asdict(spec),
                "hugging_face": hf_payload,
                "kaggle": kaggle_payload,
            }
        )

    return {"datasets": report}


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True)
