"""
Train Final Diagnosis Classifier (BCC, SCC, No Malignancy)

This script trains a weakly supervised slide-level classifier using 
pre-extracted WSI patch features (CTransPath embeddings). Patch features 
are aggregated using a Perceiver Resampler with positional encoding to 
predict slide-level diagnosis.

Architecture:
- Positional MLP for spatial encoding
- Perceiver Resampler for feature aggregation
- Linear classification head (3 classes)

Classes:
- Basal Cell Carcinoma (BCC)
- Squamous Cell Carcinoma (SCC)
- No Malignancy

Author: Solmaz Haddady
Date: 03.04.2026
"""



import os, math, json, random, h5py, argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from datasets.dataset import SlideClassificationDataset
from training.utils import collate_pad, seed_all
from models.perceiver import PerceiverResamplerClassifier


# -----------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size" ,type=int , defult=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()
# ------------------------


#BATCH_SIZE   = 1
#EPOCHS       = 10
#LR           = 1e-4
WEIGHT_DECAY = 0.1
BETAS        = (0.9, 0.95)
EPS          = 1e-8
TMAX         = EPOCHS
MAX_PATCHES  = 2000      # cap patches per slide to avoid OOM (keeps Perceiver size unchanged)
VAL_FRACTION = 0.20
SEED         = 42
LABEL_COLUMN = "FINAL_Diagnosis"

# ------------------------
# Train / Eval
# ------------------------
def run_epoch(model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for step, (feats, mask, labels) in enumerate(loader):
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):     # automatic mixed precision (AMP)
            logits = model(feats, mask)
            loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        if (step == 0) and train:
            # one-time debug
            print("Batch shapes -> feats:", tuple(feats.shape), "mask:", tuple(mask.shape))
    avg_loss = total_loss / max(1,total)
    acc = correct / max(1,total)
    return avg_loss, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset
    ds = SlideClassificationDataset(CSV_PATH, H5_DIRS, label_column=LABEL_COLUMN)
    N = len(ds)
    val_size = int(round(N * VAL_FRACTION))
    train_size = N - val_size
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_pad)

    # model/opt/sched
    model = PerceiverResamplerClassifier(num_classes=len(ds.label2idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=BETAS, eps=EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TMAX)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # resume if exists
    best_val_acc, start_epoch = 0.0, 0
    if os.path.isfile(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch  = ckpt.get("epoch", -1) + 1
        print(f"[resume] epoch={start_epoch} best_val_acc={best_val_acc:.4f}")

    # train
    for epoch in range(start_epoch, EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, scaler, device, train=False)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        # save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            Path(os.path.dirname(SAVE_PATH)).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": best_val_acc,
                "label2idx": ds.label2idx,
                "idx2label": ds.idx2label,
                "config": {
                    "max_patches": MAX_PATCHES,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "label_column": LABEL_COLUMN,
                }
            }, SAVE_PATH)
            print(f"[best] saved -> {SAVE_PATH} (val_acc={best_val_acc:.4f})")

if __name__ == "__main__":
    args = parse_args()
    CSV_PATH = args.csv_path
    H5_DIRS = args.h5_dirs
    SAVE_PATH = args.save_path
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    print("Configuration:")
    print(vars(args))
    main()




