'''
This sataset is creat for subtype training   for  bcc and scc 
Author :Solmaz Haddady 
'''

from collections import Counter

class SlideCriticalMultiLabel(Dataset):
    def __init__(self, df, h5_dirs):
        self.df = df.reset_index(drop=True)

        # index h5 by basename == Image_Nr
        self.h5_index = {}
        missing_dirs = 0
        for d in h5_dirs:
            if not os.path.isdir(d):
                missing_dirs += 1
                continue
            for fn in os.listdir(d):
                if fn.endswith(".h5"):
                    self.h5_index[os.path.splitext(fn)[0]] = os.path.join(d, fn)
        if missing_dirs:
            print(f"[warn] {missing_dirs} H5 dirs missing")

        # build sample list only for rows that have an h5
        self.samples = []
        missing = 0
        for _, row in self.df.iterrows():
            img = str(row[ID_COL])
            h5p = self.h5_index.get(img)
            if not h5p:
                missing += 1
                continue

            # labels (multi-label vectors)
            yb = row[BCC_COLS].values.astype(np.float32) if BCC_COLS else np.zeros(0, np.float32)
            ys = row[SCC_COLS].values.astype(np.float32) if SCC_COLS else np.zeros(0, np.float32)

            # robust head routing
            fd_raw = str(row[FD_COL])
            fd = fd_raw.strip().lower()
            if ("basal cell carcinoma" in fd) or ("bcc" in fd):
                head = "bcc"
            elif ("squamous cell carcinoma" in fd) or ("scc" in fd) or ("plattenepithel" in fd):
                head = "scc"
            elif ("no malignancy" in fd) or ("normal" in fd) or ("kein tumor" in fd) or ("benign" in fd):
                head = "normal"
            else:
                head = "normal"  # default: no subtype loss

            self.samples.append({"img": img, "h5": h5p, "head": head, "yb": yb, "ys": ys})

        print(f"[index] usable: {len(self.samples)} | missing h5: {missing}")
        hc = Counter(s["head"] for s in self.samples)
        print(f"[heads] bcc={hc.get('bcc',0)}  scc={hc.get('scc',0)}  normal={hc.get('normal',0)}")

        # simple counts for pos_weight (from samples, no H5 IO)
        bcc_stack = np.stack([s["yb"] for s in self.samples if s["head"]=="bcc"], axis=0) if any(s["head"]=="bcc" for s in self.samples) else np.zeros((0,len(BCC_COLS)))
        scc_stack = np.stack([s["ys"] for s in self.samples if s["head"]=="scc"], axis=0) if any(s["head"]=="scc" for s in self.samples) else np.zeros((0,len(SCC_COLS)))
        self.bcc_pos = bcc_stack.sum(0) if bcc_stack.size else np.zeros(len(BCC_COLS))
        self.scc_pos = scc_stack.sum(0) if scc_stack.size else np.zeros(len(SCC_COLS))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        with h5py.File(s["h5"], "r") as f:
            feats = torch.tensor(f["feats"][:], dtype=torch.float32)  #
        mask = torch.ones(feats.shape[0], dtype=torch.bool)
        return feats, mask, s["head"], torch.from_numpy(s["yb"]).float(), torch.from_numpy(s["ys"]).float(), s["img"]


def collate_pad(batch):
    B = len(batch)
    lens = [b[0].shape[0] for b in batch]
    Tm, D = max(lens), batch[0][0].shape[1]
    feats = torch.zeros(B, Tm, D); mask = torch.zeros(B, Tm, dtype=torch.bool)
    heads, yb, ys, ids = [], [], [], []
    for i,(x,m,h,bb,ss,img) in enumerate(batch):
        n = x.shape[0]; feats[i,:n]=x; mask[i,:n]=m
        heads.append(h); yb.append(bb); ys.append(ss); ids.append(img)
    return feats, mask, heads, torch.stack(yb), torch.stack(ys), ids
