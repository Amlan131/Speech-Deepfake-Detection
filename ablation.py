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
from sklearn.metrics import roc_auc_score, roc_curve
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DATA_ROOT  = "/root/amlan/demons/snlp/data/LA"
MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 8
EPOCHS     = 3        # 3 epochs is enough for ablation
LR         = 3e-5
MAX_LEN    = 64000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABLATION_CONFIGS = [
    {"name": "linear_probe",   "freeze_layers": 12, "desc": "All 12 layers frozen"},
    {"name": "freeze_bottom9", "freeze_layers": 9,  "desc": "Bottom 9 frozen (best config)"},
    {"name": "full_finetune",  "freeze_layers": 0,  "desc": "No layers frozen"},
]

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = Path(f"outputs/ablation_{RUN_ID}")
PLOT_DIR = OUT_DIR / "plots"
LOG_DIR  = OUT_DIR / "logs"
for d in [OUT_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ablation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.info(f"Device: {DEVICE} | Run: {RUN_ID}")

# ══════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════
def parse_protocol(path):
    records = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            records.append((parts[1], 1 if parts[-1] == "bonafide" else 0))
    return records

class ASVspoofDataset(Dataset):
    def __init__(self, records, audio_dir, processor):
        self.records   = records
        self.audio_dir = audio_dir
        self.processor = processor

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        file_id, label = self.records[idx]
        path = os.path.join(self.audio_dir, f"{file_id}.flac")
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] < MAX_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_LEN - waveform.shape[0]))
        else:
            waveform = waveform[:MAX_LEN]
        inputs = self.processor(waveform.numpy(), sampling_rate=16000,
                                return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.float32)

# ══════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════
class DeepfakeDetector(nn.Module):
    def __init__(self, freeze_layers=9):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME)
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

    def forward(self, x):
        return self.classifier(
            self.wav2vec2(x).last_hidden_state.mean(dim=1)
        ).squeeze(-1)

# ══════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx] * 100)

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(inputs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        correct    += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Eval ", leave=False):
            with autocast():
                scores = torch.sigmoid(model(inputs.to(DEVICE))).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    return roc_auc_score(all_labels, all_scores), compute_eer(all_labels, all_scores)

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    train_records = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    dev_records   = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    eval_records  = parse_protocol(f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    train_ds = ASVspoofDataset(train_records, f"{DATA_ROOT}/ASVspoof2019_LA_train/flac", processor)
    dev_ds   = ASVspoofDataset(dev_records,   f"{DATA_ROOT}/ASVspoof2019_LA_dev/flac",   processor)
    eval_ds  = ASVspoofDataset(eval_records,  f"{DATA_ROOT}/ASVspoof2019_LA_eval/flac",  processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_results  = []
    summary_rows = []

    for cfg in ABLATION_CONFIGS:
        name   = cfg["name"]
        freeze = cfg["freeze_layers"]
        desc   = cfg["desc"]
        log.info(f"\n{'='*55}")
        log.info(f"CONFIG: {name} | {desc}")
        log.info(f"{'='*55}")

        model     = DeepfakeDetector(freeze_layers=freeze).to(DEVICE)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        log.info(f"Trainable: {trainable:,} / {total:,} params")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scaler    = GradScaler()

        best_eer   = float("inf")
        best_auc   = 0
        cfg_history = []
        t_start    = time.time()

        for epoch in range(1, EPOCHS + 1):
            loss, acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
            auc, eer  = evaluate(model, dev_loader)
            elapsed   = time.time() - t_start
            log.info(f"  Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.4f} "
                     f"| Dev AUC: {auc:.4f} | Dev EER: {eer:.2f}%")
            cfg_history.append({"epoch": epoch, "loss": loss, "acc": acc,
                                 "auc": auc, "eer": eer, "config": name})
            if eer < best_eer:
                best_eer = eer
                best_auc = auc
                torch.save(model.state_dict(), OUT_DIR / f"best_{name}.pt")

        # Final eval on test set
        model.load_state_dict(torch.load(OUT_DIR / f"best_{name}.pt"))
        test_auc, test_eer = evaluate(model, eval_loader)
        total_time = time.time() - t_start

        log.info(f"\n  ✅ {name} FINAL | Test AUC: {test_auc:.4f} | Test EER: {test_eer:.2f}%")

        summary_rows.append({
            "config":          name,
            "description":     desc,
            "freeze_layers":   freeze,
            "trainable_params": trainable,
            "best_dev_eer":    round(best_eer, 2),
            "best_dev_auc":    round(best_auc, 4),
            "test_eer":        round(test_eer, 2),
            "test_auc":        round(test_auc, 4),
            "train_time_min":  round(total_time / 60, 1)
        })
        all_results.extend(cfg_history)

    # ── Save results ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    pd.DataFrame(all_results).to_csv(LOG_DIR / "all_epoch_metrics.csv", index=False)

    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    log.info("\n\n" + "="*55)
    log.info("ABLATION SUMMARY")
    log.info("="*55)
    log.info(summary_df.to_string(index=False))

    # ── Plot 1: Dev EER across epochs per config ──
    df = pd.DataFrame(all_results)
    colors = {"linear_probe": "#C44E52", "freeze_bottom9": "#55A868", "full_finetune": "#4C72B0"}
    plt.figure(figsize=(10, 5))
    for cfg_name in df["config"].unique():
        sub = df[df["config"] == cfg_name]
        plt.plot(sub["epoch"], sub["eer"], marker="o", linewidth=2,
                 color=colors[cfg_name], label=cfg_name)
    plt.xlabel("Epoch"); plt.ylabel("Dev EER (%)")
    plt.title("Ablation Study — Dev EER per Epoch", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "ablation_eer_curve.png", dpi=150); plt.close()

    # ── Plot 2: Final test EER bar chart ──
    plt.figure(figsize=(8, 5))
    bar_colors = [colors[r["config"]] for _, r in summary_df.iterrows()]
    bars = plt.bar(summary_df["config"], summary_df["test_eer"], color=bar_colors)
    plt.bar_label(bars, fmt="%.2f%%", fontsize=11, padding=3, fontweight="bold")
    plt.ylabel("Test EER (%) — Lower is Better")
    plt.title("Ablation: Test EER by Freeze Configuration", fontweight="bold")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(PLOT_DIR / "ablation_test_eer.png", dpi=150); plt.close()

    # ── Plot 3: Trainable params vs Test EER ──
    plt.figure(figsize=(7, 5))
    for _, row in summary_df.iterrows():
        plt.scatter(row["trainable_params"] / 1e6, row["test_eer"],
                    s=200, color=colors[row["config"]], zorder=5)
        plt.annotate(row["config"], (row["trainable_params"] / 1e6, row["test_eer"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
    plt.xlabel("Trainable Parameters (M)"); plt.ylabel("Test EER (%)")
    plt.title("Parameters vs Performance Tradeoff", fontweight="bold")
    plt.grid(alpha=0.3)
    plt.savefig(PLOT_DIR / "params_vs_eer.png", dpi=150); plt.close()

    log.info(f"\n✅ All ablation outputs saved to: {OUT_DIR}")
    for p in sorted(OUT_DIR.rglob("*")):
        log.info(f"   {p.relative_to(OUT_DIR)}")
