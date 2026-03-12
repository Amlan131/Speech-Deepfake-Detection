import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from pathlib import Path

WAV2VEC_DIR  = Path("outputs/run_20260311_112721")
BASELINE_DIR = Path("outputs/baseline_20260311_113828")

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_idx] * 100), float(thresholds[eer_idx])

def fix_per_attack_eer(out_dir):
    pred_path = out_dir / "eval_predictions.csv"
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}"); return

    df = pd.read_csv(pred_path)

    # Separate bonafide pool
    bonafide_df = df[df["label"] == 1]
    spoof_df    = df[df["label"] == 0]

    print(f"\nBonafide samples : {len(bonafide_df)}")
    print(f"Spoof samples    : {len(spoof_df)}")
    print(f"Attack types     : {sorted(spoof_df['attack'].unique())}\n")

    rows = []
    for atk in sorted(spoof_df["attack"].unique()):
        atk_df = spoof_df[spoof_df["attack"] == atk]

        # Combine THIS attack's spoof + ALL bonafide → compute EER
        combined    = pd.concat([atk_df, bonafide_df])
        labels_atk  = combined["label"].values
        scores_atk  = combined["score"].values

        eer, _ = compute_eer(labels_atk, scores_atk)
        rows.append({"attack": atk, "eer": round(eer, 2), "count": len(atk_df)})
        print(f"  {atk:5s} | {len(atk_df):5d} samples | EER: {eer:.2f}%")

    plot_df = pd.DataFrame(rows).sort_values("eer")
    plot_df.to_csv(out_dir / "per_attack_eer.csv", index=False)

    plt.figure(figsize=(12, 5))
    colors = ["#C44E52" if eer > 5 else "#F5A623" if eer > 1 else "#55A868"
              for eer in plot_df["eer"]]
    bars = plt.bar(plot_df["attack"], plot_df["eer"], color=colors)
    plt.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=2)
    plt.xlabel("Attack Type"); plt.ylabel("EER (%)")
    plt.title(f"Per-Attack EER (vs Bonafide Pool) — {out_dir.name}", fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "plots" / "per_attack_eer.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"\nSaved: {out_path}")

fix_per_attack_eer(WAV2VEC_DIR)
fix_per_attack_eer(BASELINE_DIR)
