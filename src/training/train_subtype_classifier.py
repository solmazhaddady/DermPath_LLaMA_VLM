"""
Train Subtype Classifier, (Critical Diagnosis)(BCC / SCC Multi-Label)

This script trains a slide-level multi-label classifier for BCC and SCC
histopathology subtypes using pre-extracted WSI patch features 
(CTransPath embeddings). Patch features are aggregated using a shared 
Perceiver backbone, followed by two independent multi-label heads.

Architecture:
- Positional MLP for spatial encoding
- Perceiver Resampler backbone (shared)
- BCC multi-label classification head
- SCC multi-label classification head

Loss:
- BCEWithLogitsLoss (multi-label)

Author: Solmaz Haddady
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score

from models.perceiver import PerceiverResampler
from datasets.subtype_dataset import SubtypeDataset


# ------------------------
# Args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()

# ----------------------------
# Hyperparams
# ----------------------------
SEED     = 1337
BATCH    = 1
EPOCHS   = 20
LR       = 3e-4
WD       = 0.01
VAL_FRAC = 0.20
NUM_WORKERS = 4

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ------------------------
# Collate
# ------------------------
def collate_fn(batch):
    B = len(batch)
    max_len = max(x[0].shape[0] for x in batch)
    dim = batch[0][0].shape[1]

    feats = torch.zeros(B, max_len, dim)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    heads, labels = [], []

    for i, (f, m, h, l) in enumerate(batch):
        n = f.shape[0]
        feats[i, :n] = f
        mask[i, :n] = True
        heads.append(h)
        labels.append(l)

    return feats, mask, heads, torch.stack(labels)


# ------------------------
# Model 
#same backbone as FD 
# ------------------------
class SubtypeClassifier(nn.Module):
    def __init__(self, num_bcc, num_scc):
        super().__init__()
        self.backbone = PerceiverResampler()

        self.head_bcc = nn.Linear(1536, num_bcc)
        self.head_scc = nn.Linear(1536, num_scc)

    def forward(self, feats, mask):
        z = self.backbone(feats).mean(dim=2).squeeze(1)

        return self.head_bcc(z), self.head_scc(z)


# ------------------------
# Metric
# ------------------------
def compute_f1(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ------------------------
# Train
# ------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Config:", vars(args))

    df = pd.read_csv(args.csv_path)
    dataset = SubtypeDataset(df, args.h5_dirs)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    num_bcc = len(dataset.bcc_cols)
    num_scc = len(dataset.scc_cols)

    model = SubtypeClassifier(num_bcc, num_scc).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        for feats, mask, heads, labels in train_loader:
            feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)

            logits_bcc, logits_scc = model(feats, mask)

            loss = 0
            for i, h in enumerate(heads):
                if h == "bcc":
                    loss += criterion(logits_bcc[i], labels[i])
                elif h == "scc":
                    loss += criterion(logits_scc[i], labels[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for feats, mask, heads, labels in val_loader:
                feats = feats.to(device)
                logits_bcc, logits_scc = model(feats, mask)

                for i, h in enumerate(heads):
                    if h == "bcc":
                        pred = torch.sigmoid(logits_bcc[i]).cpu().numpy()
                    else:
                        pred = torch.sigmoid(logits_scc[i]).cpu().numpy()

                    y_pred.append(pred)
                    y_true.append(labels[i].numpy())

        f1 = compute_f1(np.vstack(y_true), np.vstack(y_pred))
        print(f"Epoch {epoch+1}: F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model")


if __name__ == "__main__":
    main()
