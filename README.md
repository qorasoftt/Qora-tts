# Qora-TTS

Cloud-first Kinyarwanda TTS research pipeline.

## Non-negotiables

- Training runs on Kaggle GPU.
- Datasets are fetched cloud-to-cloud from Hugging Face inside Kaggle.
- Large audio files do not get downloaded to this local machine.
- Secrets do not go into Git history, notebooks, or committed config files.
- MbazaNLP Kinyarwanda TTS is the target dataset.
- WAXAL is inspection-only until transfer learning proves it helps.

## Phase 1

1. Inspect Hugging Face dataset metadata and configs without downloading audio.
2. Inspect Kaggle for existing mirrors and publishing targets.
3. Build a Kaggle probe job that authenticates with `HF_TOKEN`, validates dataset access,
   and samples metadata only.
4. Keep training code blocked until dataset inspection and a smoke run pass.

## Local commands

```powershell
python -m pip install -e .[dev]
python scripts/inspect_datasets.py --config configs/datasets.yaml
python scripts/scaffold_kaggle_probe.py --output-dir kaggle/probe
ruff check .
mypy src
pytest
```

## Kaggle secrets

- `HF_TOKEN`: Hugging Face token with read access to gated datasets and write access for uploads
  if the training job will publish artifacts.

## Current repo

- GitHub remote: `https://github.com/qorasoftt/Qora-tts`
- Local work stays authoritative; GitHub is the remote backup and collaboration surface.
