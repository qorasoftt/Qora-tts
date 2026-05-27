from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None


REPO_ID = "mbazaNLP/kinyarwanda-tts-model"
OUTPUT_DIR = Path("/kaggle/working/mbaza_kinyarwanda_tts_model")
REPORT_PATH = Path("/kaggle/working/mbaza_model_probe_report.json")
ALLOW_PATTERNS = ["*.pt", "README.md"]


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


def resolve_host(host: str) -> dict[str, str]:
    try:
        return {"host": host, "address": socket.gethostbyname(host)}
    except Exception as exc:
        return {"host": host, "error": f"{type(exc).__name__}: {exc}"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def main() -> None:
    token = read_secret("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    report: dict[str, object] = {
        "repo_id": REPO_ID,
        "hf_token_present": token is not None,
        "dns": [resolve_host("huggingface.co")],
    }

    info = api.model_info(REPO_ID, token=token)
    report["model_info"] = {
        "private": info.private,
        "downloads": info.downloads,
        "likes": info.likes,
        "sha": info.sha,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
        local_dir=OUTPUT_DIR,
        allow_patterns=ALLOW_PATTERNS,
    )

    files = collect_files(OUTPUT_DIR)
    report["output_dir"] = str(OUTPUT_DIR)
    report["allow_patterns"] = ALLOW_PATTERNS
    report["files"] = files
    report["file_count"] = len(files)
    report["total_size_bytes"] = sum(int(item["size_bytes"]) for item in files)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print({"report_path": str(REPORT_PATH)})


if __name__ == "__main__":
    main()
