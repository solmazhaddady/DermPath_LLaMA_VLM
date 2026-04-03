"""
Train Subtype Classifier (BCC / SCC multi-label)

This script trains a multi-label classifier for histopathology subtypes
using pre-extracted WSI features (CTransPath embeddings).

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


# ------------------------
# Args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


# ------------------------
# Dataset
# ------------------------
class SubtypeDataset(Dataset):
    def __init__(self, df, h5_dirs):
        self.df = df.reset_index(drop=True)

        # index h5 files
        self.h5_map = {}
        for d in h5_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".h5"):
                    self.h5_map[f.replace(".h5", "")] = os.path.join(d, f)

        # subtype columns
        self.bcc_cols = [c for c in df.columns if c.startswith("bcc_")]
        self.scc_cols = [c for c in df.columns if c.startswith("scc_")]

        self.samples = []
        for _, row in df.iterrows():
            img_id = str(row["Image_Nr"])
            if img_id not in self.h5_map:
                continue

            fd = str(row["FINAL_Diagnosis"]).lower()

            if "bcc" in fd:
                head = "bcc"
                labels = row[self.bcc_cols].values.astype(np.float32)
            elif "scc" in fd:
                head = "scc"
                labels = row[self.scc_cols].values.astype(np.float32)
            else:
                continue  # skip normal slides

            self.samples.append({
                "h5": self.h5_map[img_id],
                "head": head,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import h5py

        s = self.samples[idx]
        with h5py.File(s["h5"], "r") as f:
            feats = torch.tensor(f["feats"][:], dtype=torch.float32)

        mask = torch.ones(feats.shape[0], dtype=torch.bool)
        labels = torch.tensor(s["labels"], dtype=torch.float32)

        return feats, mask, s["head"], labels


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
