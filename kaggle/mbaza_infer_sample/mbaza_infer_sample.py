from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

TEXT = """
Itangazo ry'iduka ry'imyambaro n'inkweto.

Ukeneye kugaragara neza kandi wiyubashye?
Dusange mu iduka ryacu, tuguhitiremo imyambaro n'inkweto bikubereye.

Dufite imyambaro myiza y'abagabo, abagore n'abana,
hamwe n'inkweto ziramba kandi zifite igiciro cyiza.
Waba ugiye ku kazi, mu birori, mu rugendo cyangwa mu buzima bwa buri munsi, dufite ibigukwiriye.

Gura ibyiza. Ambara neza. Garagara neza.
Dusange uyu munsi, uhitemo ibikubereye.
""".strip()

REPO_ROOT = Path("/kaggle/temp/DeepLearningExamples")
FASTPITCH_DIR = REPO_ROOT / "PyTorch" / "SpeechSynthesis" / "FastPitch"
MODEL_CACHE_DIR = Path("/kaggle/temp/mbaza_model")
OUTPUT_DIR = Path("/kaggle/working/mbaza_infer_sample")
INPUT_PATH = OUTPUT_DIR / "input.txt"
REPORT_PATH = OUTPUT_DIR / "report.json"
WAV_PATH = OUTPUT_DIR / "audio_0.wav"
MP3_PATH = OUTPUT_DIR / "mbaza_sample.mp3"
ALLOW_PATTERNS = [
    "README.md",
    "kinyarwanda_fastpitch_checkpoint_2000.pt",
    "nvidia_waveglow256pyt_fp16.pt",
]


def run(command: list[str], cwd: Path | None = None) -> dict[str, object]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd) if cwd is not None else None,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "elapsed_seconds": round(time.time() - started, 3),
    }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return " ".join(lines)


def require_ok(step: str, result: dict[str, object]) -> None:
    if int(result["returncode"]) != 0:
        payload = {"step": step, "result": result}
        raise RuntimeError(json.dumps(payload, indent=2))


def file_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def choose_accelerator() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"mode": "cpu", "reason": "cuda_unavailable"}
    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    payload: dict[str, object] = {
        "mode": "cuda",
        "device_name": device_name,
        "capability": capability,
    }
    if capability[0] < 7:
        payload["mode"] = "cpu"
        payload["reason"] = "cuda_capability_unsupported_by_runtime"
    return payload


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    normalized_text = normalize_text(TEXT)
    INPUT_PATH.write_text(normalized_text + "\n", encoding="utf-8")

    report: dict[str, object] = {
        "normalized_text": normalized_text,
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "accelerator": choose_accelerator(),
        "steps": [],
    }

    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    clone_result = run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/NVIDIA/DeepLearningExamples.git",
            str(REPO_ROOT),
        ]
    )
    report["steps"].append({"name": "clone_repo", "result": clone_result})
    require_ok("clone_repo", clone_result)

    install_result = run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=FASTPITCH_DIR,
    )
    report["steps"].append({"name": "install_requirements", "result": install_result})
    require_ok("install_requirements", install_result)

    model_dir = Path(
        snapshot_download(
            repo_id="mbazaNLP/kinyarwanda-tts-model",
            repo_type="model",
            local_dir=MODEL_CACHE_DIR,
            allow_patterns=ALLOW_PATTERNS,
        )
    )
    report["model_dir"] = str(model_dir)
    report["model_files"] = [
        file_info(model_dir / "kinyarwanda_fastpitch_checkpoint_2000.pt"),
        file_info(model_dir / "nvidia_waveglow256pyt_fp16.pt"),
        file_info(model_dir / "README.md"),
    ]

    infer_cmd = [
        sys.executable,
        "inference.py",
        "--fastpitch",
        str(model_dir / "kinyarwanda_fastpitch_checkpoint_2000.pt"),
        "--waveglow",
        str(model_dir / "nvidia_waveglow256pyt_fp16.pt"),
        "--input",
        str(INPUT_PATH),
        "--output",
        str(OUTPUT_DIR),
        "--batch-size",
        "1",
        "--denoising-strength",
        "0.0",
    ]
    if report["accelerator"]["mode"] == "cuda":
        infer_cmd[2:2] = ["--cuda", "--amp"]
    infer_result = run(infer_cmd, cwd=FASTPITCH_DIR)
    report["steps"].append({"name": "run_inference", "result": infer_result})
    require_ok("run_inference", infer_result)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(WAV_PATH),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(MP3_PATH),
    ]
    ffmpeg_result = run(ffmpeg_cmd)
    report["steps"].append({"name": "convert_mp3", "result": ffmpeg_result})
    require_ok("convert_mp3", ffmpeg_result)

    report["wav"] = file_info(WAV_PATH)
    report["mp3"] = file_info(MP3_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
