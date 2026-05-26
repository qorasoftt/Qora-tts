# Dataset Findings

Date: 2026-05-26

## Verified from local inspection

- `mbazaNLP/kinyarwanda-tts-dataset`
  - Hugging Face access path exists.
  - The dataset is gated.
  - The current token can resolve metadata, but config inspection is blocked until the
    authenticated Hugging Face account is granted access on the dataset page.
  - Kaggle search did not reveal a reliable Mbaza mirror.

- `google/WaxalNLP`
  - Hugging Face access path exists and is open.
  - Config inspection succeeded without downloading audio.
  - Kaggle search did not reveal an existing mirror.

## Implication

The cloud-to-cloud path is:

1. Kaggle notebook authenticates with `HF_TOKEN`.
2. Kaggle notebook pulls datasets directly from Hugging Face.
3. Kaggle notebook writes checkpoints and evaluation samples back to Hugging Face and/or a
   Kaggle dataset.

If you insist that the training datasets themselves must live on Kaggle before training, that is a
separate mirroring step. It cannot be completed for Mbaza until the Hugging Face account is
approved for gated access.
