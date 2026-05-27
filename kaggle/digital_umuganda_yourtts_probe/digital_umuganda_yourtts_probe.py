from __future__ import annotations

import json
import os
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
Waba ugiye ku kazi, mu birori, mu rugendo cyangwa mu buzima bwa buri munsi,
dufite ibigukwiriye.

Gura ibyiza. Ambara neza. Garagara neza.
Dusange uyu munsi, uhitemo ibikubereye.
""".strip()

MODEL_REPO = "DigitalUmuganda/Kinyarwanda_YourTTS_v1"
MODEL_CACHE_DIR = Path("/kaggle/temp/digital_umuganda_yourtts_v1")
OUTPUT_DIR = Path("/kaggle/working/digital_umuganda_yourtts_probe")
REPORT_PATH = OUTPUT_DIR / "report.json"
WAV_PATH = OUTPUT_DIR / "audio.wav"
MP3_PATH = OUTPUT_DIR / "yourtts_sample.mp3"
ALLOW_PATTERNS = [
    "README.md",
    "best_model.pth",
    "SE_checkpoint.pth.tar",
    "config.json",
    "config_se.json",
    "conditioning_audio.wav",
    "speakers.pth",
]
TTS_PIN = (
    "git+https://github.com/coqui-ai/TTS@"
    "0910cb76bcd85df56bf43654bb31427647cdfd0d#egg=TTS"
)
MICROMAMBA_BIN = Path("/kaggle/temp/bin/micromamba")
PY310_PREFIX = Path("/kaggle/temp/yourtts-py310")
HELPER_SCRIPT = OUTPUT_DIR / "run_yourtts_helper.py"
HELPER_REPORT_PATH = OUTPUT_DIR / "helper_report.json"


def run(command: list[str]) -> dict[str, object]:
    started = time.time()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
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


def file_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def install_micromamba(report: dict[str, object]) -> None:
    MICROMAMBA_BIN.parent.mkdir(parents=True, exist_ok=True)
    bash_command = (
        "mkdir -p /kaggle/temp/bin "
        "&& curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xvj -C /kaggle/temp/bin --strip-components=1 bin/micromamba"
    )
    result = run(["bash", "-lc", bash_command])
    report["steps"].append({"name": "install_micromamba", "result": result})
    require_ok("install_micromamba", result)


def setup_python310_env(report: dict[str, object]) -> Path:
    create_env = run(
        [
            str(MICROMAMBA_BIN),
            "create",
            "-y",
            "-p",
            str(PY310_PREFIX),
            "-c",
            "conda-forge",
            "python=3.10",
            "pip",
        ]
    )
    report["steps"].append({"name": "create_py310_env", "result": create_env})
    require_ok("create_py310_env", create_env)

    env_python = PY310_PREFIX / "bin" / "python"
    install_tts = run([str(env_python), "-m", "pip", "install", TTS_PIN])
    report["steps"].append({"name": "install_tts_py310", "result": install_tts})
    require_ok("install_tts_py310", install_tts)
    return env_python


def write_helper_script(
    model_dir: Path,
    normalized_text: str,
    accelerator: dict[str, object],
) -> None:
    use_cuda = accelerator["mode"] == "cuda"
    helper_source = f"""
from __future__ import annotations

import json
import os
from pathlib import Path

if {str(not use_cuda)}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import soundfile as sf
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer

model_dir = Path({str(model_dir)!r})
output_wav = Path({str(WAV_PATH)!r})
report_path = Path({str(HELPER_REPORT_PATH)!r})
text = {normalized_text!r}
use_cuda = {str(use_cuda)}

config = load_config(str(model_dir / "config.json"))
synthesizer = Synthesizer(
    str(model_dir / "best_model.pth"),
    str(model_dir / "config.json"),
    tts_speakers_file=str(model_dir / "speakers.pth"),
    encoder_checkpoint=str(model_dir / "SE_checkpoint.pth.tar"),
    encoder_config=str(model_dir / "config_se.json"),
    use_cuda=use_cuda,
)

audio = synthesizer.tts(text, speaker_wav=str(model_dir / "conditioning_audio.wav"))
if isinstance(audio, list):
    audio = np.asarray(audio, dtype=np.float32)
sample_rate = getattr(config.audio, "sample_rate", 24000)
sf.write(output_wav, audio, samplerate=sample_rate, format="WAV")

characters = getattr(config, "characters", None)
report = {{
    "audio_sample_rate": sample_rate,
    "use_phonemes": getattr(config, "use_phonemes", None),
    "phonemizer": getattr(config, "phonemizer", None),
    "text_cleaner": getattr(config, "text_cleaner", None),
    "characters": characters.__dict__ if characters is not None else None,
}}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
"""
    HELPER_SCRIPT.write_text(helper_source.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    normalized_text = normalize_text(TEXT)
    accelerator = choose_accelerator()
    if accelerator["mode"] != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    report: dict[str, object] = {
        "repo_id": MODEL_REPO,
        "normalized_text": normalized_text,
        "python": sys.version,
        "accelerator": accelerator,
        "steps": [],
    }

    install_micromamba(report)
    env_python = setup_python310_env(report)

    model_dir = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir=MODEL_CACHE_DIR,
            allow_patterns=ALLOW_PATTERNS,
        )
    )
    report["model_dir"] = str(model_dir)
    report["model_files"] = [file_info(model_dir / pattern) for pattern in ALLOW_PATTERNS]

    write_helper_script(model_dir, normalized_text, accelerator)
    started = time.time()
    helper_result = run([str(env_python), str(HELPER_SCRIPT)])
    report["steps"].append({"name": "run_helper_inference", "result": helper_result})
    require_ok("run_helper_inference", helper_result)
    report["inference_seconds"] = round(time.time() - started, 3)
    report["config"] = json.loads(HELPER_REPORT_PATH.read_text(encoding="utf-8"))

    ffmpeg_result = run(
        [
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
    )
    report["steps"].append({"name": "convert_mp3", "result": ffmpeg_result})
    require_ok("convert_mp3", ffmpeg_result)

    report["wav"] = file_info(WAV_PATH)
    report["mp3"] = file_info(MP3_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
