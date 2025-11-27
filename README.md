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

├── anli_best_results/ # Saved model + metrics + plots.
├── serve/ # Deployment or inference scripts
├── Dockerfile
├── requirements.txt
├── ANLI_Finetuning.ipynb # Main notebook
└── README.md

## 3. Notebook Highlights

The notebook is organized into clear sections that mirror a standard experimental pipeline:

### 0. Configuration

- Choose model name
- Batch size, gradient accumulation, learning rate, number of epochs
- GPU/CPU auto-detect

### 1. Reproducibility

All relevant seeds are fixed (numpy, Python, Torch, CUDA).

### 2. Data Loading + EDA

- Loads ANLI from the Hugging Face hub
- Calculates dataset sizes and label distribution
- Stores a small EDA summary JSON

### 3. Tokenization + Preprocessing
Everything is tokenized with ```AutoTokenizer``` and converted into a format ready for PyTorch + Trainer.

### 4. Model Setup
```AutoModelForSequenceClassification``` with ```num_labels=3```.

### 5. Metrics
The evaluation uses accuracy, label-specific F1 scores, and macro-F1 (which is the metric used for checkpoint selection).

### 6–8. Training Setup & Execution
- TrainingArguments tuned for larger models
- Early stopping
- Logging every 50 steps
- Saves best model automatically

### 9–11. Evaluation & Plots
- Full dev/test evaluation
- Confusion matrix (PNG + JSON)
- Classification report
- Training curves for loss and macro-F1

### 12. Reproducibility Notes
Saves configuration values and pointers to logs.

## 4. Running Locally (No Docker)
- Install Python 3.10
- Install dependencies:
```
pip install -r requirements.txt
```
- Make sure your environment has a GPU + CUDA (optional but recommended).
- Open the notebook:
```
jupyter notebook
```
- Run all cells.
Artifacts appear automatically under ```anli_best_results/```.

## 5. Using the Docker Image
A simple Dockerfile is included. It installs dependencies, copies the saved model artifacts, and launches a notebook server inside the container.

**Build the image**
```
docker build -t anli-finetune .
```
**Run the container**
```
docker run -it -p 8080:8080 anli-finetune
```
Jupyter Notebook will be available at:
```
http://localhost:8080
```
This setup is useful if you want a clean environment for experiments or if you want to share the notebook + results with someone without requiring them to install the dependencies manually.

**Note**
The Dockerfile uses the lightweight python:3.10-slim base and installs only minimal OS-level packages needed for building Python dependencies. GPU support inside Docker would require building an image on top of an NVIDIA CUDA base image, which is optional depending on your deployment plan.

## 6. Training Tips
Because ANLI is fairly large and RoBERTa-large is memory-hungry, a few hyperparameters tend to matter:
- **BATCH_SIZE:**
If you run out of GPU memory, reduce BATCH_SIZE and increase GRAD_ACC.
- **MAX_LENGTH:**
ANLI examples can be long. 256 works well as a balance between speed and coverage.
- **USE_ALL_ROUNDS:**
Training on all rounds significantly increases training time. If you're prototyping, use only R2.
- **Warmup + Weight Decay:**
Both help stabilize training for large models, especially on adversarial datasets.

## 7. Example Results (Typical Trends)
Depending on the GPU and hyperparameters, you should expect:
- Training F1 improving gradually due to adversarial difficulty
- Macro-F1 ~ mid 40s to mid 50s for RoBERTa-large (across all rounds)
- Confusion matrix showing imbalance in contradiction vs neutral

Actual values will be stored in:
```
anli_best_results/metrics.json
```

## 8. Reproducibility
To reproduce a run:
- Set the same seed
- Use identical hyperparameters
- Keep the same training/evaluation order
- Use the same HF model version

All of this is automatically stored in:
```
anli_best_results/reproducibility.json
```
This file captures the critical configuration fields from the training session.
