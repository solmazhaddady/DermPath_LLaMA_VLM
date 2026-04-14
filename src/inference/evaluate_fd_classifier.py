"""
Evaluate Final Diagnosis Model (BCC, SCC, No Malignancy)

- Loads trained model
- Runs inference on validation set
- Saves predictions and confusion matrix

Author: Solmaz Haddady
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from datasets.slide_dataset import SlideClassificationDataset
from training.utils import collate_pad
from models.perceiver import PerceiverResamplerClassifier


# ------------------------
# Args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


# ------------------------
# Plot Confusion Matrix
# ------------------------
def plot_cm(cm, labels, path, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val:.2f}" if normalize else int(val),
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running inference...")
    print(vars(args))

    # dataset
    ds = SlideClassificationDataset(args.csv_path, args.h5_dirs)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_pad
    )

    # model
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model = PerceiverResamplerClassifier(num_classes=len(ckpt["label2idx"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx2label = ckpt["idx2label"]
    label_names = list(idx2label.values())

    softmax = nn.Softmax(dim=1)

    y_true, y_pred, probs_all = [], [], []

    with torch.no_grad():
        for feats, mask, labels in tqdm(loader):
            feats = feats.to(device)
            logits = model(feats, mask)

            probs = softmax(logits).cpu().numpy()[0]
            pred = logits.argmax(dim=1).item()

            y_true.append(idx2label[labels.item()])
            y_pred.append(idx2label[pred])
            probs_all.append(probs)

    # save predictions
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred
    })

    for i, name in enumerate(label_names):
        df[f"prob_{name}"] = [p[i] for p in probs_all]

    csv_path = os.path.join(args.output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)

    print("Saved:", csv_path)

    # metrics
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    print("\nAccuracy:", np.mean(np.array(y_true) == np.array(y_pred)))
    print("\n", classification_report(y_true, y_pred))

    # plots
    plot_cm(cm, label_names,
            os.path.join(args.output_dir, "cm_counts.png"),
            normalize=False)

    plot_cm(cm, label_names,
            os.path.join(args.output_dir, "cm_normalized.png"),
            normalize=True)


if __name__ == "__main__":
    main()
