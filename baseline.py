import os, json, logging, warnings, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from sklearn.manifold import TSNE
import joblib
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DATA_ROOT = "/root/amlan/demons/snlp/data/LA"
N_MFCC    = 40
DURATION  = 4.0

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = Path(f"outputs/baseline_{RUN_ID}")
PLOT_DIR = OUT_DIR / "plots"
LOG_DIR  = OUT_DIR / "logs"

for d in [OUT_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "baseline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.info(f"Run ID     : {RUN_ID}")
log.info(f"Output dir : {OUT_DIR}")

# Save config
config = {
    "run_id": RUN_ID, "model": "MFCC+SVM",
    "n_mfcc": N_MFCC, "duration_sec": DURATION,
    "svm_kernel": "rbf", "svm_C": 1.0,
    "data_root": DATA_ROOT
}
with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# ══════════════════════════════════════════════
#  PROTOCOL PARSER
# ══════════════════════════════════════════════
def parse_protocol(path):
    records = []
    with open(path) as f:
        for line in f:
            parts  = line.strip().split()
            attack = parts[3] if parts[3] != "-" else "bonafide"
            label  = 1 if parts[-1] == "bonafide" else 0
            records.append((parts[1], label, attack))
    return records

# ══════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════
def extract_features(file_id, audio_dir, n_mfcc=N_MFCC, duration=DURATION):
    path    = os.path.join(audio_dir, f"{file_id}.flac")
    y, sr   = librosa.load(path, sr=16000, duration=duration)
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    # Mean + Std pooling over time for each feature type
    feat = np.concatenate([
        mfcc.mean(axis=1),   mfcc.std(axis=1),
        delta.mean(axis=1),  delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1)
    ])
    return feat  # 240-dim vector

def build_features(records, audio_dir, split_name):
    X, y, attacks, failed = [], [], [], []
    for file_id, label, attack in tqdm(records, desc=f"Extracting [{split_name}]"):
        try:
            feat = extract_features(file_id, audio_dir)
            X.append(feat); y.append(label); attacks.append(attack)
        except Exception as e:
            failed.append(file_id)
    if failed:
        log.warning(f"{len(failed)} files failed to load in [{split_name}]")
    return np.array(X), np.array(y), attacks

# ══════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_idx] * 100), float(thresholds[eer_idx])

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
def plot_roc(labels, scores, auc):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.4f}")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — MFCC+SVM Baseline", fontweight="bold")
    plt.legend(fontsize=12); plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150); plt.close()
    log.info("Saved: roc_curve.png")

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Spoof","Bonafide"],
                yticklabels=["Spoof","Bonafide"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix — MFCC+SVM", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150); plt.close()
    log.info("Saved: confusion_matrix.png")

def plot_score_distribution(labels, scores):
    labels  = np.array(labels)
    scores  = np.array(scores)
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels==1], bins=60, alpha=0.65,
             color="#55A868", label="Bonafide", density=True)
    plt.hist(scores[labels==0], bins=60, alpha=0.65,
             color="#C44E52", label="Spoof",    density=True)
    plt.xlabel("SVM Decision Score"); plt.ylabel("Density")
    plt.title("Score Distribution — MFCC+SVM Baseline", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "score_distribution.png", dpi=150); plt.close()
    log.info("Saved: score_distribution.png")

def plot_per_attack_eer(labels, scores, attacks):
    attack_types = sorted(set(attacks))
    rows = []
    for atk in attack_types:
        idx = [i for i, a in enumerate(attacks) if a == atk]
        if len(set(np.array(labels)[idx])) < 2: continue
        eer, _ = compute_eer(np.array(labels)[idx], np.array(scores)[idx])
        rows.append({"attack": atk, "eer": round(eer, 2), "count": len(idx)})
    df = pd.DataFrame(rows).sort_values("eer")
    df.to_csv(OUT_DIR / "per_attack_eer.csv", index=False)
    colors = ["#55A868" if r["attack"]=="bonafide" else "#C44E52"
              for _, r in df.iterrows()]
    plt.figure(figsize=(10, 4))
    plt.bar(df["attack"], df["eer"], color=colors)
    plt.xlabel("Attack Type"); plt.ylabel("EER (%)")
    plt.title("Per-Attack EER — MFCC+SVM Baseline", fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "per_attack_eer.png", dpi=150); plt.close()
    log.info("Saved: per_attack_eer.png")

def plot_mfcc_mean_comparison(X_eval, y_eval):
    y_eval = np.array(y_eval)
    # First 40 dims are MFCC means
    real_mean  = X_eval[y_eval==1, :N_MFCC].mean(axis=0)
    spoof_mean = X_eval[y_eval==0, :N_MFCC].mean(axis=0)
    x = np.arange(N_MFCC)
    plt.figure(figsize=(10, 4))
    plt.plot(x, real_mean,  marker="o", color="#55A868", label="Bonafide", linewidth=2)
    plt.plot(x, spoof_mean, marker="o", color="#C44E52", label="Spoof",    linewidth=2)
    plt.fill_between(x, real_mean, spoof_mean, alpha=0.1, color="gray")
    plt.xlabel("MFCC Coefficient Index"); plt.ylabel("Mean Value")
    plt.title("Average MFCC Profile: Bonafide vs Spoof", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "mfcc_mean_comparison.png", dpi=150); plt.close()
    log.info("Saved: mfcc_mean_comparison.png")

def plot_tsne(X, y, attacks):
    log.info("Running t-SNE on 400 eval samples (takes ~1 min)...")
    idx    = np.random.choice(len(X), min(400, len(X)), replace=False)
    X_sub  = X[idx]; y_sub = np.array(y)[idx]
    proj   = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_sub)
    plt.figure(figsize=(7, 5))
    for cls, name, col in [(1,"Bonafide","#55A868"),(0,"Spoof","#C44E52")]:
        m = y_sub == cls
        plt.scatter(proj[m,0], proj[m,1], label=name,
                    alpha=0.65, c=col, s=22, edgecolors="none")
    plt.title("t-SNE: MFCC Features — Bonafide vs Spoof", fontweight="bold")
    plt.legend(); plt.axis("off")
    plt.savefig(PLOT_DIR / "tsne_mfcc.png", dpi=150); plt.close()
    log.info("Saved: tsne_mfcc.png")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    # ── Extract features ──
    train_recs = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    eval_recs  = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    log.info(f"Train samples: {len(train_recs)} | Eval samples: {len(eval_recs)}")

    X_train, y_train, _            = build_features(train_recs, f"{DATA_ROOT}/ASVspoof2019_LA_train/flac", "train")
    X_eval,  y_eval,  eval_attacks = build_features(eval_recs,  f"{DATA_ROOT}/ASVspoof2019_LA_eval/flac",  "eval")

    log.info(f"Feature shape — Train: {X_train.shape} | Eval: {X_eval.shape}")

    # Save raw features for reproducibility
    np.save(LOG_DIR / "X_train.npy", X_train)
    np.save(LOG_DIR / "X_eval.npy",  X_eval)
    np.save(LOG_DIR / "y_train.npy", y_train)
    np.save(LOG_DIR / "y_eval.npy",  y_eval)

    # ── Scale ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval  = scaler.transform(X_eval)
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")

    # ── Train SVM ──
    log.info("Training SVM (this may take 5–10 min on full data)...")
    t_svm = time.time()
    svm   = SVC(kernel="rbf", probability=True, C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    svm_time = time.time() - t_svm
    log.info(f"SVM trained in {svm_time:.1f}s")
    joblib.dump(svm, OUT_DIR / "svm_model.pkl")

    # ── Evaluate ──
    scores = svm.predict_proba(X_eval)[:, 1]
    preds  = svm.predict(X_eval)
    auc    = roc_auc_score(y_eval, scores)
    eer, thresh = compute_eer(y_eval, scores)
    report = classification_report(y_eval, preds,
                                   target_names=["Spoof","Bonafide"],
                                   output_dict=True)

    log.info(f"\n🏆 BASELINE RESULT | AUC: {auc:.4f} | EER: {eer:.2f}%")
    log.info("\n" + classification_report(y_eval, preds,
                                          target_names=["Spoof","Bonafide"]))

    # ── Save all outputs ──
    pred_df = pd.DataFrame({
        "label": y_eval, "score": scores,
        "pred": preds, "attack": eval_attacks
    })
    pred_df.to_csv(OUT_DIR / "eval_predictions.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report.csv")

    total_time = time.time() - t_start
    final_metrics = {
        "run_id": RUN_ID, "model": "MFCC+SVM",
        "feature_dim": X_train.shape[1],
        "test_auc":           round(auc,   4),
        "test_eer_pct":       round(eer,   2),
        "best_threshold":     round(thresh,4),
        "precision_bonafide": round(report["Bonafide"]["precision"], 4),
        "recall_bonafide":    round(report["Bonafide"]["recall"],    4),
        "f1_bonafide":        round(report["Bonafide"]["f1-score"],  4),
        "precision_spoof":    round(report["Spoof"]["precision"],    4),
        "recall_spoof":       round(report["Spoof"]["recall"],       4),
        "f1_spoof":           round(report["Spoof"]["f1-score"],     4),
        "svm_train_time_sec": round(svm_time,   1),
        "total_time_min":     round(total_time/60, 2)
    }
    with open(OUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # ── All plots ──
    plot_roc(y_eval, scores, auc)
    plot_confusion_matrix(y_eval, preds)
    plot_score_distribution(y_eval, scores)
    plot_per_attack_eer(y_eval, scores, eval_attacks)
    plot_mfcc_mean_comparison(X_eval, y_eval)
    plot_tsne(X_eval, y_eval, eval_attacks)

    # ── Final folder summary ──
    log.info(f"\n All outputs saved to: {OUT_DIR}")
    log.info("📁 Structure:")
    for p in sorted(OUT_DIR.rglob("*")):
        log.info(f"   {p.relative_to(OUT_DIR)}")
