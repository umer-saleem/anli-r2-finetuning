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

