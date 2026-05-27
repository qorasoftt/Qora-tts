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

WORD_PAIRS = [
    ("itangazo", "itangaazo"),
    ("umwana", "umwaana"),
    ("abana", "abaana"),
    ("umuntu", "umuuntu"),
    ("abantu", "abaantu"),
    ("inka", "iinka"),
    ("imbwa", "iimbwa"),
    ("ingabo", "ingaabo"),
    ("igihugu", "igihuugu"),
    ("umurimo", "umuriimo"),
    ("ibikorwa", "ibikoorwa"),
    ("ikigo", "ikiigo"),
    ("icyumba", "icyuumba"),
    ("igitabo", "igitaabo"),
    ("umunsi", "umuunsi"),
    ("ameza", "ameeza"),
    ("umusozi", "umusozi"),
    ("isoko", "isoóko"),
    ("amafaranga", "amafaraanga"),
    ("ubukwe", "ubuukwe"),
    ("urugo", "uruugo"),
    ("ikiraro", "ikiraaro"),
    ("umuryango", "umuryaango"),
    ("indirimbo", "indiriimbo"),
    ("amagambo", "amagaambo"),
    ("inkuru", "inkuuru"),
    ("ikibazo", "ikibaazo"),
    ("igisubizo", "igisuubizo"),
    ("umuyobozi", "umuyoobozi"),
    ("ubuyobozi", "ubuyoobozi"),
]

MODEL_REPO = "DigitalUmuganda/Kinyarwanda_YourTTS_v1"
MODEL_CACHE_DIR = Path("/kaggle/temp/digital_umuganda_yourtts_v1")
OUTPUT_DIR = Path("/kaggle/working/group1_hinted_probe")
REPORT_PATH = OUTPUT_DIR / "report.json"
HELPER_REPORT_PATH = OUTPUT_DIR / "helper_report.json"
HELPER_SCRIPT = OUTPUT_DIR / "run_group1_hinted_helper.py"
MP3_DIR = OUTPUT_DIR / "mp3"
WAV_DIR = OUTPUT_DIR / "wav"
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
    return " ".join(part.strip() for part in normalized.splitlines() if part.strip())


def safe_stem(text: str) -> str:
    normalized = normalize_text(text)
    sanitized = []
    for char in normalized:
        if char.isalnum():
            sanitized.append(char.lower())
        else:
            sanitized.append("_")
    stem = "".join(sanitized).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem


def require_ok(step: str, result: dict[str, object]) -> None:
    if int(result["returncode"]) != 0:
        raise RuntimeError(json.dumps({"step": step, "result": result}, indent=2))


def choose_accelerator() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"mode": "cpu", "reason": "cuda_unavailable"}
    capability = torch.cuda.get_device_capability(0)
    payload: dict[str, object] = {
        "mode": "cuda",
        "device_name": torch.cuda.get_device_name(0),
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
    result = run(
        [
            "bash",
            "-lc",
            "mkdir -p /kaggle/temp/bin "
            "&& curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
            "| tar -xvj -C /kaggle/temp/bin --strip-components=1 bin/micromamba",
        ]
    )
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


def write_helper_script(model_dir: Path, accelerator: dict[str, object]) -> None:
    use_cuda = accelerator["mode"] == "cuda"
    normalized_pairs = [(normalize_text(w), normalize_text(s)) for w, s in WORD_PAIRS]
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

pairs = {normalized_pairs!r}
model_dir = Path({str(model_dir)!r})
wav_dir = Path({str(WAV_DIR)!r})
report_path = Path({str(HELPER_REPORT_PATH)!r})
use_cuda = {str(use_cuda)}

wav_dir.mkdir(parents=True, exist_ok=True)
config = load_config(str(model_dir / "config.json"))
synthesizer = Synthesizer(
    str(model_dir / "best_model.pth"),
    str(model_dir / "config.json"),
    tts_speakers_file=str(model_dir / "speakers.pth"),
    encoder_checkpoint=str(model_dir / "SE_checkpoint.pth.tar"),
    encoder_config=str(model_dir / "config_se.json"),
    use_cuda=use_cuda,
)

items = []
for index, (written, spoken) in enumerate(pairs, start=1):
    audio = synthesizer.tts(spoken, speaker_wav=str(model_dir / "conditioning_audio.wav"))
    if isinstance(audio, list):
        audio = np.asarray(audio, dtype=np.float32)
    sample_rate = getattr(config.audio, "sample_rate", 24000)
    stem = "{{:02d}}_{{}}_written__{{}}".format(index, safe_stem(spoken), safe_stem(written))
    wav_path = wav_dir / f"{{stem}}.wav"
    sf.write(wav_path, audio, samplerate=sample_rate, format="WAV")
    items.append({{
        "index": index,
        "written": written,
        "spoken_hint": spoken,
        "wav_path": str(wav_path),
        "wav_size_bytes": wav_path.stat().st_size,
    }})

characters = getattr(config, "characters", None)
report = {{
    "audio_sample_rate": getattr(config.audio, "sample_rate", 24000),
    "use_phonemes": getattr(config, "use_phonemes", None),
    "phonemizer": getattr(config, "phonemizer", None),
    "text_cleaner": getattr(config, "text_cleaner", None),
    "characters": characters.__dict__ if characters is not None else None,
    "items": items,
}}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
"""
    helper_source = (
        "from __future__ import annotations\n\n"
        "def safe_stem(text: str) -> str:\n"
        "    sanitized = []\n"
        "    for char in text:\n"
        "        if char.isalnum():\n"
        "            sanitized.append(char.lower())\n"
        "        else:\n"
        "            sanitized.append('_')\n"
        "    stem = ''.join(sanitized).strip('_')\n"
        "    while '__' in stem:\n"
        "        stem = stem.replace('__', '_')\n"
        "    return stem\n\n"
        + helper_source[len("from __future__ import annotations\n\n") :]
    )
    HELPER_SCRIPT.write_text(helper_source.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MP3_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    accelerator = choose_accelerator()
    if accelerator["mode"] != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    report: dict[str, object] = {
        "repo_id": MODEL_REPO,
        "pair_count": len(WORD_PAIRS),
        "pairs": [
            {"written": normalize_text(written), "spoken_hint": normalize_text(spoken)}
            for written, spoken in WORD_PAIRS
        ],
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

    write_helper_script(model_dir, accelerator)
    started = time.time()
    helper_result = run([str(env_python), str(HELPER_SCRIPT)])
    report["steps"].append({"name": "run_group1_hinted_helper", "result": helper_result})
    require_ok("run_group1_hinted_helper", helper_result)
    report["inference_seconds"] = round(time.time() - started, 3)

    helper_report = json.loads(HELPER_REPORT_PATH.read_text(encoding="utf-8"))
    report["config"] = {
        "audio_sample_rate": helper_report["audio_sample_rate"],
        "use_phonemes": helper_report["use_phonemes"],
        "phonemizer": helper_report["phonemizer"],
        "text_cleaner": helper_report["text_cleaner"],
        "characters": helper_report["characters"],
    }

    mp3_items: list[dict[str, object]] = []
    for item in helper_report["items"]:
        wav_path = Path(item["wav_path"])
        mp3_path = MP3_DIR / (wav_path.stem + ".mp3")
        ffmpeg_result = run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                str(mp3_path),
            ]
        )
        require_ok(f"convert_mp3_{wav_path.stem}", ffmpeg_result)
        mp3_items.append(
            {
                "index": item["index"],
                "written": item["written"],
                "spoken_hint": item["spoken_hint"],
                "wav": file_info(wav_path),
                "mp3": file_info(mp3_path),
            }
        )

    report["items"] = mp3_items
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
