# High-Performance ANLI Finetuning

This repository contains an end-to-end pipeline for finetuning **RoBERTa-large** on the **ANLI (Adversarial NLI)** dataset using Hugging Face Transformers.
The project started as an attempt to build a repeatable finetuning setup that includes:
- Data loading and lightweight EDA.
- Tokenization and preprocessing.
- GPU-accelerated training with Trainer.
- Automatic metric tracking.
- Saving all artifacts (model, tokenizer, plots, logs, metrics).
- Optional Docker deployment.

The overall workflow follows the typical structure of a research-oriented experiment, with an emphasis on reproducibility and keeping track of artifacts.

## 1. Project Overview
ANLI is a challenging adversarial NLI benchmark composed of three rounds (R1, R2, R3).
By default, the notebook trains on **all rounds**, but this can be changed to **only Round 2 (R2)** if you want a faster experiment:

```
USE_ALL_ROUNDS = True   # set to False to use only R2
```

