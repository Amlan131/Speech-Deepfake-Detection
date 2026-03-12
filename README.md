# Speech Deepfake Detection using wav2vec2

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ICASSP](https://img.shields.io/badge/Cited-ICASSP%202026-orange)](https://arxiv.org/abs/2603.01482)

A speech deepfake detection system built on fine-tuned **wav2vec2** 
self-supervised representations, trained and evaluated on the 
**ASVspoof 2019 Logical Access (LA)** benchmark dataset.

Submitted as part of the Speech and Natural Language Processing 
course project.

---

## Results Summary

| Model | Test AUC | Test EER | 
|---|---|---|
| MFCC + SVM (Baseline) | 0.9565 | 9.79% |
| **wav2vec2 + FCNN (Ours)** | **0.9836** | **2.97%** |
| ICASSP 2026 SOTA | — | 0.22% |

**3.3× improvement** over the handcrafted feature baseline.

---

## Ablation Study

| Configuration | Trainable Params | Dev EER | Test EER |
|---|---|---|---|
| Linear Probe (Freeze All) | 9.5M | 0.77% | 5.60% |
| **Freeze Bottom 9 (Ours)** | **30.7M** | **0.44%** | **2.97%** |
| Full Fine-tune | 94.5M | — | Collapsed (AUC=0.50) |

Key finding: Full fine-tuning causes **catastrophic forgetting** 
due to the mismatch between wav2vec2's pretraining objective and 
the binary classification task.

---

## Architecture

