"""
Dataset used for train subtype classifier from BCC and SCC head 
Author : Solmaz Haddady 

"""

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

