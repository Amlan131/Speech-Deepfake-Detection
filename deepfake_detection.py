import os, json, logging, warnings, time
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.metrics import (roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DATA_ROOT  = "/root/amlan/demons/snlp/data/LA"
MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 8
EPOCHS     = 5
LR         = 3e-5
MAX_LEN    = 64000
DEVICE     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = Path(f"outputs/run_{RUN_ID}")
CKPT_DIR   = OUT_DIR / "checkpoints"
PLOT_DIR   = OUT_DIR / "plots"
LOG_DIR    = OUT_DIR / "logs"

for d in [OUT_DIR, CKPT_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.info(f"Run ID     : {RUN_ID}")
log.info(f"Output dir : {OUT_DIR}")
log.info(f"Device     : {DEVICE}")

# Save config
config = {
    "run_id": RUN_ID, "model": MODEL_NAME, "batch_size": BATCH_SIZE,
    "epochs": EPOCHS, "lr": LR, "max_len": MAX_LEN,
    "device": str(DEVICE), "freeze_layers": 9,
    "data_root": DATA_ROOT
}
with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# ══════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════
def parse_protocol(protocol_path):
    records = []
    with open(protocol_path) as f:
        for line in f:
            parts   = line.strip().split()
            file_id = parts[1]
            attack  = parts[3] if parts[3] != "-" else "bonafide"
            label   = 1 if parts[-1] == "bonafide" else 0
            records.append((file_id, label, attack))
    return records

class ASVspoofDataset(Dataset):
    def __init__(self, records, audio_dir, processor, max_len=MAX_LEN):
        self.records   = records
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_len   = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        file_id, label, attack = self.records[idx]
        path = os.path.join(self.audio_dir, f"{file_id}.flac")
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_len]
        inputs = self.processor(waveform.numpy(), sampling_rate=16000,
                                return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.float32), attack

def collate_fn(batch):
    inputs  = torch.stack([b[0] for b in batch])
    labels  = torch.stack([b[1] for b in batch])
    attacks = [b[2] for b in batch]
    return inputs, labels, attacks

# ══════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════
class DeepfakeDetector(nn.Module):
    def __init__(self, model_name, freeze_layers=9):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        hidden = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),    nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_values):
        out    = self.wav2vec2(input_values).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

    def get_embedding(self, input_values):
        with torch.no_grad():
            return self.wav2vec2(input_values).last_hidden_state.mean(dim=1)

# ══════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_idx] * 100), float(thresholds[eer_idx])

# ══════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss, n_correct, n_total = 0, 0, 0
    batch_log = []
    for step, (inputs, labels, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch} Train")):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(inputs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds     = (torch.sigmoid(logits) > 0.5).float()
        n_correct += (preds == labels).sum().item()
        n_total   += labels.size(0)
        total_loss += loss.item()
        batch_log.append({"step": step, "loss": round(loss.item(), 5)})

    avg_loss = total_loss / len(loader)
    acc      = n_correct / n_total
    pd.DataFrame(batch_log).to_csv(LOG_DIR / f"epoch{epoch}_batch_loss.csv", index=False)
    return avg_loss, acc

def evaluate(model, loader, split_name="eval"):
    model.eval()
    all_labels, all_scores, all_attacks, all_preds = [], [], [], []
    with torch.no_grad():
        for inputs, labels, attacks in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            inputs = inputs.to(DEVICE)
            with autocast():
                logits = model(inputs)
            scores = torch.sigmoid(logits).cpu().numpy()
            preds  = (scores > 0.5).astype(float)
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_attacks.extend(attacks)

    all_labels  = np.array(all_labels)
    all_scores  = np.array(all_scores)
    all_preds   = np.array(all_preds)
    auc         = roc_auc_score(all_labels, all_scores)
    eer, thresh = compute_eer(all_labels, all_scores)
    return auc, eer, thresh, all_labels, all_scores, all_preds, all_attacks

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
def plot_training_curves(history):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, col, title, color in zip(
        axes,
        ["loss", "acc", "auc", "eer"],
        ["Training Loss", "Training Accuracy", "Dev AUC", "Dev EER (%) ↓"],
        ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
    ):
        ax.plot(df["epoch"], df[col], marker="o", color=color, linewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150); plt.close()
    log.info("Saved: training_curves.png")

def plot_roc(labels, scores, auc):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.4f}")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Speech Deepfake Detection", fontweight="bold")
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
    plt.title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150); plt.close()
    log.info("Saved: confusion_matrix.png")

def plot_score_distribution(labels, scores):
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 1], bins=60, alpha=0.65, color="#55A868",
             label="Bonafide", density=True)
    plt.hist(scores[labels == 0], bins=60, alpha=0.65, color="#C44E52",
             label="Spoof",    density=True)
    plt.xlabel("Detection Score"); plt.ylabel("Density")
    plt.title("Score Distribution: Bonafide vs Spoof", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "score_distribution.png", dpi=150); plt.close()
    log.info("Saved: score_distribution.png")

def plot_per_attack_eer(labels, scores, attacks):
    attack_types = sorted(set(attacks))
    rows = []
    for atk in attack_types:
        idx  = [i for i, a in enumerate(attacks) if a == atk]
        if len(set(np.array(labels)[idx])) < 2: continue
        eer, _ = compute_eer(np.array(labels)[idx], np.array(scores)[idx])
        rows.append({"attack": atk, "eer": round(eer, 2), "count": len(idx)})
    df = pd.DataFrame(rows).sort_values("eer")
    df.to_csv(OUT_DIR / "per_attack_eer.csv", index=False)
    plt.figure(figsize=(10, 4))
    colors = ["#C44E52" if r["attack"] != "bonafide" else "#55A868" for _, r in df.iterrows()]
    plt.bar(df["attack"], df["eer"], color=colors)
    plt.xlabel("Attack Type"); plt.ylabel("EER (%)")
    plt.title("Per-Attack EER — Lower is Better", fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "per_attack_eer.png", dpi=150); plt.close()
    log.info("Saved: per_attack_eer.png")

def plot_tsne(model, eval_ds):
    model.eval()
    embeddings, tsne_labels = [], []
    loader = DataLoader(eval_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    with torch.no_grad():
        for inputs, lbs, _ in loader:
            emb = model.get_embedding(inputs.to(DEVICE)).cpu().numpy()
            embeddings.append(emb)
            tsne_labels.extend(lbs.numpy())
            if len(tsne_labels) >= 300: break
    embeddings  = np.vstack(embeddings)[:300]
    tsne_labels = np.array(tsne_labels)[:300]
    log.info("Running t-SNE (this takes ~1 min)...")
    proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(7, 5))
    for cls, name, col in [(1,"Bonafide","#55A868"), (0,"Spoof","#C44E52")]:
        m = tsne_labels == cls
        plt.scatter(proj[m,0], proj[m,1], label=name, alpha=0.65, c=col, s=22, edgecolors="none")
    plt.title("t-SNE: Real vs Fake Speech Embeddings", fontweight="bold")
    plt.legend(); plt.axis("off")
    plt.savefig(PLOT_DIR / "tsne_embeddings.png", dpi=150); plt.close()
    log.info("Saved: tsne_embeddings.png")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    train_records = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    dev_records   = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    eval_records  = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    log.info(f"Train: {len(train_records)} | Dev: {len(dev_records)} | Eval: {len(eval_records)}")

    train_ds = ASVspoofDataset(train_records, f"{DATA_ROOT}/ASVspoof2019_LA_train/flac", processor)
    dev_ds   = ASVspoofDataset(dev_records,   f"{DATA_ROOT}/ASVspoof2019_LA_dev/flac",   processor)
    eval_ds  = ASVspoofDataset(eval_records,  f"{DATA_ROOT}/ASVspoof2019_LA_eval/flac",  processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model     = DeepfakeDetector(MODEL_NAME, freeze_layers=9).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    best_eer = float("inf")
    history  = []
    epoch_metrics = []

    # ── Training loop ──
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        auc, eer, thresh, _, _, _, _ = evaluate(model, dev_loader, "dev")
        scheduler.step()

        elapsed = time.time() - t_ep
        log.info(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | Acc: {train_acc:.4f} "
                 f"| Dev AUC: {auc:.4f} | Dev EER: {eer:.2f}% | Time: {elapsed:.1f}s")

        row = {"epoch": epoch, "loss": round(loss,5), "acc": round(train_acc,4),
               "auc": round(auc,4), "eer": round(eer,2), "thresh": round(thresh,4),
               "lr": scheduler.get_last_lr()[0], "time_sec": round(elapsed,1)}
        history.append(row)
        epoch_metrics.append(row)

        # Save checkpoint every epoch
        ckpt_path = CKPT_DIR / f"epoch{epoch}_eer{eer:.2f}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "eer": eer, "auc": auc}, ckpt_path)

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            log.info(f"  ✅ New best model saved (EER={eer:.2f}%)")

    # Save epoch log
    pd.DataFrame(epoch_metrics).to_csv(LOG_DIR / "epoch_metrics.csv", index=False)

    # ── Final evaluation ──
    log.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt"))
    auc, eer, thresh, labels, scores, preds, attacks = evaluate(model, eval_loader, "test")

    # Classification report
    report = classification_report(labels, preds, target_names=["Spoof","Bonafide"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report.csv")
    log.info("\n" + classification_report(labels, preds, target_names=["Spoof","Bonafide"]))

    # Save raw predictions
    pred_df = pd.DataFrame({"label": labels, "score": scores,
                             "pred": preds, "attack": attacks})
    pred_df.to_csv(OUT_DIR / "eval_predictions.csv", index=False)

    # Final metrics JSON
    total_time = time.time() - t_start
    final_metrics = {
        "run_id": RUN_ID, "test_auc": round(auc, 4),
        "test_eer_pct": round(eer, 2), "best_threshold": round(thresh, 4),
        "precision_bonafide": round(report["Bonafide"]["precision"], 4),
        "recall_bonafide":    round(report["Bonafide"]["recall"], 4),
        "f1_bonafide":        round(report["Bonafide"]["f1-score"], 4),
        "precision_spoof":    round(report["Spoof"]["precision"], 4),
        "recall_spoof":       round(report["Spoof"]["recall"], 4),
        "f1_spoof":           round(report["Spoof"]["f1-score"], 4),
        "total_training_time_min": round(total_time / 60, 2)
    }
    with open(OUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    log.info(f"\n🏆 FINAL | AUC: {auc:.4f} | EER: {eer:.2f}%")

    # ── All plots ──
    plot_training_curves(history)
    plot_roc(labels, scores, auc)
    plot_confusion_matrix(labels, preds)
    plot_score_distribution(labels, scores)
    plot_per_attack_eer(labels, scores, attacks)
    plot_tsne(model, eval_ds)

    log.info(f"\n✅ Everything saved to: {OUT_DIR}")
    log.info(f"📁 Structure:")
    for p in sorted(OUT_DIR.rglob("*")):
        log.info(f"   {p.relative_to(OUT_DIR)}")
