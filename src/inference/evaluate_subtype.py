"""
Evaluate Subtype Classification (BCC / SCC multi-label)

- Computes macro-F1
- Saves predictions

Author: Solmaz Haddady
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score

from training.dataset_subtype import SubtypeDataset
from training.train_subtype import SubtypeClassifier


# ------------------------
# Args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


# ------------------------
# Metric
# ------------------------
def compute_f1(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running subtype evaluation...")
    print(vars(args))

    df = pd.read_csv(args.csv_path)
    dataset = SubtypeDataset(df, args.h5_dirs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # model
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    num_bcc = len(dataset.bcc_cols)
    num_scc = len(dataset.scc_cols)

    model = SubtypeClassifier(num_bcc, num_scc)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for feats, mask, heads, labels in loader:
            feats = feats.to(device)

            logits_bcc, logits_scc = model(feats, mask)

            if heads[0] == "bcc":
                pred = torch.sigmoid(logits_bcc).cpu().numpy()[0]
            else:
                pred = torch.sigmoid(logits_scc).cpu().numpy()[0]

            y_pred.append(pred)
            y_true.append(labels.numpy()[0])

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    f1 = compute_f1(y_true, y_pred)
    print(f"Macro F1: {f1:.4f}")

    # save predictions
    df_out = pd.DataFrame(y_pred)
    Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_path, index=False)

    print("Saved predictions:", args.output_path)


if __name__ == "__main__":
    main()
