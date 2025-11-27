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
The training pipeline uses:
- **RoBERTa-large**
- **FP16 mixed precision** (if GPU available)
- **Gradient accumulation** to fit the large model on smaller GPUs
- **Early stopping**
- **Macro-F1** as the main model selection metric
- Fully reproducible random seeds

After training, the script exports:
- ```metrics.json``` (dev + test)
- ```trainer_history.json```
- ```classification_report.json```
- ```confusion_matrix.png``` and JSON
- Loss and F1 curves
- The final model + tokenizer under ```anli_best_results/model/```

All output files are written to:
```
./anli_best_results/
```

## 2. Repository Structure

├── anli_best_results/ # Saved model + metrics + plots
├── serve/ # Deployment or inference scripts
├── Dockerfile
├── requirements.txt
├── ANLI_Finetuning.ipynb # Main notebook
└── README.md


