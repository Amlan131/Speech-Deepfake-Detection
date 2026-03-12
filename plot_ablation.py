# save as plot_ablation.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path("outputs/ablation_20260311_121003")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hardcode results from your logs ──
summary = [
    {"config": "linear_probe",   "desc": "Freeze All (12/12)",
     "freeze_layers": 12, "trainable_params": 9_530_625,
     "best_dev_eer": 0.77, "test_eer": 5.60, "test_auc": 0.9828},

    {"config": "freeze_bottom9", "desc": "Freeze Bottom 9 (ours)",
     "freeze_layers": 9,  "trainable_params": 30_794_241,
     "best_dev_eer": 0.44, "test_eer": 2.97, "test_auc": 0.9836},

    {"config": "full_finetune",  "desc": "Full Fine-tune (collapsed)",
     "freeze_layers": 0,  "trainable_params": 94_585_089,
     "best_dev_eer": None, "test_eer": None,  "test_auc": 0.5000},
]

epoch_data = [
    # linear_probe
    {"config": "linear_probe",   "epoch": 1, "loss": 0.2184, "acc": 0.9591, "auc": 0.9968, "eer": 2.43},
    {"config": "linear_probe",   "epoch": 2, "loss": 0.0288, "acc": 0.9941, "auc": 0.9990, "eer": 0.91},
    {"config": "linear_probe",   "epoch": 3, "loss": 0.0153, "acc": 0.9974, "auc": 0.9997, "eer": 0.77},
    # freeze_bottom9
    {"config": "freeze_bottom9", "epoch": 1, "loss": 0.1555, "acc": 0.9610, "auc": 0.9996, "eer": 0.44},
    {"config": "freeze_bottom9", "epoch": 2, "loss": 0.0240, "acc": 0.9961, "auc": 0.9990, "eer": 0.59},
    {"config": "freeze_bottom9", "epoch": 3, "loss": 0.0183, "acc": 0.9974, "auc": 0.9996, "eer": 0.69},
    # full_finetune
    {"config": "full_finetune",  "epoch": 1, "loss": 0.7450, "acc": 0.9040, "auc": 0.5000, "eer": 50.0},
    {"config": "full_finetune",  "epoch": 2, "loss": 0.8109, "acc": 0.8983, "auc": 0.5000, "eer": 50.0},
    {"config": "full_finetune",  "epoch": 3, "loss": 0.8120, "acc": 0.8983, "auc": 0.5000, "eer": 50.0},
]

colors = {
    "linear_probe":   "#F5A623",
    "freeze_bottom9": "#55A868",
    "full_finetune":  "#C44E52"
}

df      = pd.DataFrame(epoch_data)
sum_df  = pd.DataFrame(summary)

# Save CSVs
df.to_csv(OUT_DIR / "epoch_metrics.csv", index=False)
sum_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

# ── Plot 1: Dev EER across epochs ──
plt.figure(figsize=(10, 5))
for cfg in df["config"].unique():
    sub = df[df["config"] == cfg]
    plt.plot(sub["epoch"], sub["eer"], marker="o", linewidth=2.5,
             color=colors[cfg], label=cfg)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Dev EER (%)", fontsize=12)
plt.title("Ablation Study — Dev EER per Epoch", fontweight="bold", fontsize=13)
plt.legend(fontsize=11); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ablation_eer_curve.png", dpi=150); plt.close()
print("Saved: ablation_eer_curve.png")

# ── Plot 2: Training Loss curves ──
plt.figure(figsize=(10, 5))
for cfg in df["config"].unique():
    sub = df[df["config"] == cfg]
    plt.plot(sub["epoch"], sub["loss"], marker="o", linewidth=2.5,
             color=colors[cfg], label=cfg)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Ablation Study — Training Loss per Epoch", fontweight="bold", fontsize=13)
plt.legend(fontsize=11); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ablation_loss_curve.png", dpi=150); plt.close()
print("Saved: ablation_loss_curve.png")

# ── Plot 3: Test EER bar chart (excluding collapsed) ──
valid = sum_df[sum_df["test_eer"].notna()].copy()
bar_colors = [colors[c] for c in valid["config"]]
plt.figure(figsize=(8, 5))
bars = plt.bar(valid["desc"], valid["test_eer"], color=bar_colors, width=0.5)
plt.bar_label(bars, fmt="%.2f%%", fontsize=12, padding=4, fontweight="bold")
plt.ylabel("Test EER (%) — Lower is Better", fontsize=12)
plt.title("Ablation: Final Test EER by Configuration", fontweight="bold", fontsize=13)
plt.ylim(0, max(valid["test_eer"]) * 1.3)
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(PLOT_DIR / "ablation_test_eer.png", dpi=150); plt.close()
print("Saved: ablation_test_eer.png")

# ── Plot 4: Trainable params vs Test EER ──
plt.figure(figsize=(8, 5))
for _, row in sum_df.iterrows():
    eer = row["test_eer"] if row["test_eer"] else 50.0
    col = colors[row["config"]]
    marker = "X" if row["config"] == "full_finetune" else "o"
    plt.scatter(row["trainable_params"] / 1e6, eer,
                s=250, color=col, zorder=5, marker=marker)
    label = row["desc"] + (" ← COLLAPSED" if row["config"] == "full_finetune" else "")
    plt.annotate(label,
                 (row["trainable_params"] / 1e6, eer),
                 textcoords="offset points",
                 xytext=(10, 5), fontsize=9)
plt.xlabel("Trainable Parameters (Millions)", fontsize=12)
plt.ylabel("Test EER (%)", fontsize=12)
plt.title("Parameters vs Performance Tradeoff", fontweight="bold", fontsize=13)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PLOT_DIR / "params_vs_eer.png", dpi=150); plt.close()
print("Saved: params_vs_eer.png")

# ── Plot 5: Full comparison — Baseline vs wav2vec2 vs Ablations ──
all_models = [
    {"model": "MFCC+SVM\n(Baseline)",        "test_eer": 9.79,  "color": "#4C72B0"},
    {"model": "Linear Probe\n(Freeze All)",   "test_eer": 5.60,  "color": "#F5A623"},
    {"model": "Freeze Bottom 9\n(Ours) ✅",   "test_eer": 2.97,  "color": "#55A868"},
    {"model": "Full Fine-tune\n(Collapsed)",  "test_eer": 50.0,  "color": "#C44E52"},
]
comp_df = pd.DataFrame(all_models)
plt.figure(figsize=(11, 6))
bars = plt.bar(comp_df["model"], comp_df["test_eer"],
               color=comp_df["color"], width=0.5)
plt.bar_label(bars, fmt="%.2f%%", fontsize=11, padding=4, fontweight="bold")
plt.ylabel("Test EER (%) — Lower is Better", fontsize=12)
plt.title("Full Model Comparison: Baseline → Ablation → Best", fontweight="bold", fontsize=13)
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(PLOT_DIR / "full_comparison.png", dpi=150); plt.close()
print("Saved: full_comparison.png")

print(f"\n✅ All 5 plots saved to {PLOT_DIR}")
