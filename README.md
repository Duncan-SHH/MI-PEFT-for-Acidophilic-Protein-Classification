# MI-PEFT-for-Acidophilic-Protein-Classification
# MoE-Integrated PEFT on ESM Cambrian for Acidophilic Protein Classification

This project implements an MoE-integrated PEFT framework on **ESM Cambrian (600M)** for the downstream task of **acidophilic protein classification**.

The backbone protein language model (PLM) used in this project is **ESM Cambrian / ESM++ large**, which should be downloaded from Hugging Face before running the code:

https://huggingface.co/Synthyra/ESMplusplus_large

Make sure the model is available locally or accessible from your environment before running `ind.py`.

---

## Overview

This repository is designed for binary classification of protein sequences into:

- **Acidophilic proteins**
- **Non-acidophilic proteins**

The framework combines:

- a pretrained **PLM backbone** based on ESM Cambrian,
- a **DeepSeekMoE classifier head**,
- and **PEFT-based adaptation** for efficient downstream fine-tuning.

The goal is to improve parameter efficiency while maintaining strong classification performance on the acidophilic protein prediction task.

---

## Dataset

The dataset used in this project follows the benchmark dataset reported in the original acidophilic protein prediction study, including both acidophilic and non-acidophilic protein sequences. The code directly reads these pre-processed sequence files in `Acidophilic-main`.

Expected data files include:
- `Positive.txt`
- `Negative.txt`
- `Ind-positive.txt`
- `Ind-negative.txt`


---


## Model Description

The framework uses:

1. **ESM Cambrian (600M)** as the pretrained protein language model backbone  
2. **DeepSeekMoE** as the classification head  
3. **PEFT** for efficient adaptation of the backbone during fine-tuning  

Instead of relying on a standard dense classification head, this project integrates an MoE-based head to improve flexibility and parameter efficiency for downstream sequence classification.

---

## Requirements

You need a Python environment with the required deep learning and bioinformatics dependencies installed. The code mainly relies on packages such as:

- `torch`
- `transformers`
- `peft`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

If your implementation uses additional modules such as custom MoE components, make sure the corresponding Python files are included in the same project directory.

---

## Usage

Run the main script with:

```bash
python ind.py

