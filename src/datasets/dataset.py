'''
Dataset
Author: Solmaz Haddady

'''
class SlideClassificationDataset(Dataset):
    """
    Fast version: pre-index all .h5 files, then match rows by Image_Nr.
    """
    def __init__(self, csv_path, h5_dirs, label_column="FINAL_Diagnosis", verbose=True):
        import glob
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column

        # Map class strings -> indices
        classes = sorted(self.df[label_column].dropna().unique().tolist())
        self.label2idx = {c: i for i, c in enumerate(classes)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}
        if verbose:
            print(f"[classes] {self.label2idx}", flush=True)

        # 1) Build an index of all available .h5 files once
        h5_map = {}  # Image_Nr (string) -> full path
        total_files = 0
        for d in h5_dirs:
            if not os.path.isdir(d):
                print(f"[warn] missing dir: {d}", flush=True)
                continue
            files = glob.glob(os.path.join(d, "*.h5"))
            total_files += len(files)
            for p in files:
                #  "<Image_Nr>.h5"
                key = os.path.splitext(os.path.basename(p))[0]
                h5_map[str(key)] = p
            if verbose:
                print(f"[index] {d} : {len(files)} .h5", flush=True)

        if verbose:
            print(f"[index] total .h5 files indexed: {total_files}", flush=True)

        # 2) Match CSV rows to the index (O(rows))
        self.samples = []
        missing = 0
        for i, row in self.df.iterrows():
            if (i % 1000 == 0) and verbose:
                print(f"[match] processed {i}/{len(self.df)} rows...", flush=True)

            image_nr = row.get("Image_Nr")
            label = row.get(label_column)
            if pd.isna(image_nr) or pd.isna(label):
                continue

            path = h5_map.get(str(image_nr))
            if path is None:
                missing += 1
                continue

            self.samples.append({
                "h5_path": path,
                "label_idx": self.label2idx[str(label)],
                "image_nr": str(image_nr),
            })

        if verbose:
            print(f"[match] matched: {len(self.samples)} | missing: {missing}", flush=True)

        if len(self.samples) == 0:
            raise RuntimeError("No matching .h5 files found for any CSV rows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        with h5py.File(rec["h5_path"], "r") as f:
            feats = torch.tensor(f["feats"][:]).float()   # [N,768]
            coords_raw = f["coords"][:]                   # [N,3]
        x = coords_raw[:,1]; y = coords_raw[:,2]
        x = x / (x.max() if x.max()!=0 else 1.0)
        y = y / (y.max() if y.max()!=0 else 1.0)
        coords = torch.tensor(np.stack([x,y], axis=1)).float()
        return feats, coords, rec["label_idx"]
