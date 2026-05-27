from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path


def run_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def resolve_host(host: str) -> dict[str, object]:
    try:
        return {"host": host, "address": socket.gethostbyname(host)}
    except Exception as exc:
        return {"host": host, "error": f"{type(exc).__name__}: {exc}"}


def probe_torch() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"import_error": f"{type(exc).__name__}: {exc}"}

    payload: dict[str, object] = {
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(0)
    return payload


def main() -> None:
    report = {
        "python_version": run_command(["python", "--version"]),
        "torch": probe_torch(),
        "nvidia_smi": run_command(["nvidia-smi"]) if shutil.which("nvidia-smi") else None,
        "dns": [
            resolve_host("huggingface.co"),
            resolve_host("github.com"),
            resolve_host("www.kaggle.com"),
        ],
    }

    output_path = Path("/kaggle/working/qora_tts_gpu_probe.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print({"output_path": str(output_path)})


if __name__ == "__main__":
    main()
