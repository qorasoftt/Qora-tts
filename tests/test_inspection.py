from __future__ import annotations

from pathlib import Path

from qora_tts.inspection import load_dataset_specs, parse_kaggle_csv


def test_load_dataset_specs_reads_expected_entries() -> None:
    specs = load_dataset_specs(Path("configs/datasets.yaml"))
    assert [spec.name for spec in specs] == ["mbaza_main", "waxal_pool"]
    assert specs[0].hf_repo_id == "mbazaNLP/kinyarwanda-tts-dataset"


def test_parse_kaggle_csv_handles_basic_output() -> None:
    raw = "\n".join(
        [
            "ref,title,size",
            "owner/example,Example,123",
            "owner/example-two,Example Two,456",
        ]
    )
    parsed = parse_kaggle_csv(raw)
    assert parsed == [
        {"ref": "owner/example", "title": "Example", "size": "123"},
        {"ref": "owner/example-two", "title": "Example Two", "size": "456"},
    ]
