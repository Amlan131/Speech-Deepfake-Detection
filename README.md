# Speech Deepfake Detection using wav2vec2

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ICASSP](https://img.shields.io/badge/Cited-ICASSP%202026-orange)](https://arxiv.org/abs/2603.01482)

A speech deepfake detection system built on fine-tuned **wav2vec2** self-supervised
representations, trained and evaluated on the **ASVspoof 2019 Logical Access (LA)**
benchmark dataset. Submitted as part of the Speech and Natural Language Processing
course project.

---

## Results Summary

| Model | Test AUC | Test EER |
|---|---:|---:|
| MFCC + SVM (Baseline) | 0.9565 | 9.79% |
| **wav2vec2 + FCNN (Ours)** | **0.9836** | **2.97%** |
| ICASSP 2026 SOTA | — | 0.22% |

**3.3× improvement** over the handcrafted feature baseline.

---

## Ablation Study

| Configuration | Trainable Params | Dev EER | Test EER |
|---|---:|---:|---:|
| Linear Probe (Freeze All 12) | 9.5M | 0.77% | 5.60% |
| **Freeze Bottom 9 (Ours)** | **30.7M** | **0.44%** | **2.97%** |
| Full Fine-tune | 94.5M | — | Collapsed (AUC=0.50) |

> **Key finding:** Full fine-tuning causes **catastrophic forgetting** due to the
> mismatch between wav2vec2's pretraining objective and the binary classification task.

---

## Architecture

```
Input Audio (16kHz)
       ↓
wav2vec2-base (Facebook/Meta)
  ├── 12 Transformer Encoder Layers
  ├── Bottom 9 → FROZEN (preserve acoustic representations)
  └── Top 3    → Fine-tuned (adapt to spoofing detection task)
       ↓
Mean Pooling over time  (T × 768 → 768)
       ↓
FCNN Classifier
  Linear(768 → 256) → ReLU → Dropout(0.3)
  Linear(256 → 64)  → ReLU
  Linear(64  → 1)   → Sigmoid
       ↓
Binary Output: REAL (Bonafide) / FAKE (Spoof)
```

---

## Project Structure

```
├── deepfake_detection.py           # Main wav2vec2 training pipeline
├── baseline.py                     # MFCC + SVM baseline
├── ablation.py                     # Ablation: freeze configs comparison
├── demo.py                         # Live inference on any audio file
├── fix_plots.py                    # Per-attack EER plot generator
├── plot_ablation.py                # Ablation results plot generator
├── requirements.txt
├── outputs/
│   ├── run/        # wav2vec2 main results
│   │   ├── config.json
│   │   ├── final_metrics.json
│   │   ├── classification_report.csv
│   │   ├── eval_predictions.csv
│   │   ├── per_attack_eer.csv
│   │   ├── logs/
│   │   │   ├── train.log
│   │   │   └── epoch_metrics.csv
│   │   └── plots/
│   │       ├── training_curves.png
│   │       ├── roc_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── score_distribution.png
│   │       ├── per_attack_eer.png
│   │       └── tsne_embeddings.png
│   ├── baseline/   # MFCC+SVM baseline results
│   │   ├── final_metrics.json
│   │   ├── classification_report.csv
│   │   └── plots/
│   │       ├── roc_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── score_distribution.png
│   │       └── per_attack_eer.png
│   └── ablation/   # Ablation study results
│       ├── ablation_summary.csv
│       ├── ablation_summary.json
│       └── plots/
│           ├── ablation_eer_curve.png
│           ├── ablation_loss_curve.png
│           ├── ablation_test_eer.png
│           ├── params_vs_eer.png
│           └── full_comparison.png
└── README.md
```

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/speech-deepfake-detection.git
cd speech-deepfake-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Request and download ASVspoof 2019 LA dataset:
```bash
wget -c "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
unzip LA.zip -d data/
```

Update `DATA_ROOT` in each script:
```python
DATA_ROOT = "./data/LA"
```

Expected folder structure after unzip:
```
data/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
```

---

## Training

### Main Model — wav2vec2 + FCNN
```bash
python deepfake_detection.py
```

Expected output per epoch:
```
Epoch 1/5 | Loss: 0.1354 | Acc: 0.9712 | Dev AUC: 0.9983 | Dev EER: 0.87%
Epoch 2/5 | Loss: 0.0243 | Acc: 0.9961 | Dev AUC: 0.9984 | Dev EER: 1.52%
Epoch 3/5 | Loss: 0.0112 | Acc: 0.9982 | Dev AUC: 0.9989 | Dev EER: 1.25%
Epoch 4/5 | Loss: 0.0038 | Acc: 0.9996 | Dev AUC: 0.9998 | Dev EER: 0.43%  ← Best
Epoch 5/5 | Loss: 0.0018 | Acc: 0.9996 | Dev AUC: 0.9996 | Dev EER: 0.48%
FINAL TEST | AUC: 0.9836 | EER: 2.97%
```

### Baseline — MFCC + SVM
```bash
python baseline.py
```

Expected output:
```
BASELINE RESULT | AUC: 0.9565 | EER: 9.79%
```

### Ablation Study
```bash
python ablation.py
```

---

## Inference

Run live inference on any `.flac` or `.wav` audio file:
```bash
python demo.py path/to/audio.flac
```

Example output:
```
File    : audio.flac
Score   : 0.9312  (1.0 = definitely real, 0.0 = definitely fake)
Verdict : ✅ REAL (Bonafide)
```

---

## Key Findings

### 1. SSL vs Handcrafted Features
wav2vec2 representations reduce EER by **3.3×** (9.79% → 2.97%) compared to MFCC
features. Self-supervised representations capture fine-grained spoofing artifacts —
phase inconsistencies, vocoder fingerprints, and unnatural spectral transitions —
that handcrafted features cannot detect.

### 2. Catastrophic Forgetting Under Full Fine-Tuning
Full fine-tuning of all 94.5M parameters caused training collapse (AUC = 0.50, loss
diverged from 0.74 → 0.81 across 3 epochs). Selective freezing of the bottom 9
transformer layers is critical to preserve pretrained acoustic representations.

### 3. Per-Attack Generalization Gap

| Attack | EER | Difficulty |
|---|---:|---|
| A14 | 0.26% | Easiest |
| A15 | 0.45% | Easy |
| A07 | 1.28% | Moderate |
| A12 | 6.96% | Hard |
| A18 | 12.80% | Very Hard |
| A17 | 21.39% | Hardest |

Attack **A17** (neural waveform synthesis) remains the most challenging, consistent
with findings in the ICASSP 2026 benchmark paper (arXiv:2603.01482).

### 4. Dev vs Eval Generalization
- Dev EER: **0.43%** → Test EER: **2.97%**
- Gap attributed to unseen attack types (A07–A19) in the evaluation set
  vs seen attacks (A01–A06) in the development set.

---

## Hardware and Training Time

| Component | Detail |
|---|---|
| GPU | CUDA-enabled |
| Training time | ~22 minutes (5 epochs) |
| Batch size | 8 with FP16 mixed precision |
| Total parameters | 94,585,089 |
| Trainable parameters | 30,794,241 (32.6%) |
| Optimizer | AdamW (LR = 3e-5) |
| Loss | Weighted BCE (pos_weight = 4.0) |

---

## Plots

### Training Curves
![Training Curves](outputs/runs/plots/training_curves.png)

### ROC Curve
![ROC Curve](outputs/runs/plots/roc_curve.png)

### Confusion Matrix — wav2vec2
![Confusion Matrix](outputs/runs/plots/confusion_matrix.png)

### Confusion Matrix — MFCC+SVM Baseline
![Baseline Confusion Matrix](outputs/baseline/plots/confusion_matrix.png)

### Per-Attack EER — wav2vec2
![Per-Attack EER](outputs/runs/plots/per_attack_eer.png)

### Per-Attack EER — Baseline
![Baseline Per-Attack EER](outputs/baseline/plots/per_attack_eer.png)

### t-SNE Embedding Visualization
![t-SNE](outputs/runs/plots/tsne_embeddings.png)

### Full Model Comparison
![Full Comparison](outputs/ablation/plots/full_comparison.png)

### Ablation EER Curve
![Ablation EER](outputs/ablation/plots/ablation_eer_curve.png)

### Ablation Loss Curve
![Ablation Loss](outputs/ablation/plots/ablation_loss_curve.png)

---

## References

1. **[ICASSP 2026]** "A SUPERB-Style Benchmark of Self-Supervised Speech Models
   for Audio Deepfake Detection." arXiv:2603.01482

2. **[NeurIPS 2020]** A. Baevski, H. Zhou, A. Mohamed, M. Auli.
   "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations."
   arXiv:2006.11477

3. **[IEEE TASLP 2021]** A. Nautsch et al.
   "ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized,
   Converted and Replayed Speech."

---

## License

MIT License — free to use for academic and research purposes.
