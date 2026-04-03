# ------------------------
# Seeding
# ------------------------
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all()

# ------------------------
# Collate with patch cap
# ------------------------
def uniform_downsample(feats, coords, max_patches):
    T = feats.shape[0]
    if T <= max_patches:
        return feats, coords
    idx = torch.linspace(0, T-1, steps=max_patches).long()
    return feats.index_select(0, idx), coords.index_select(0, idx)

def collate_pad(batch):
    feats_list, coords_list, labels_list = zip(*batch)
    feats_c, coords_c = [], []
    for f, c in zip(feats_list, coords_list):
        f2, c2 = uniform_downsample(f, c, MAX_PATCHES)
        feats_c.append(f2); coords_c.append(c2)
    T_max = max(f.shape[0] for f in feats_c)
    B = len(batch)
    feats_pad = torch.zeros(B, T_max, 768, dtype=torch.float32)
    mask      = torch.zeros(B, T_max, dtype=torch.bool)
    for i, f in enumerate(feats_c):
        t = f.shape[0]
        feats_pad[i,:t] = f
        mask[i,:t] = True
    labels = torch.tensor(labels_list, dtype=torch.long)
    return feats_pad, mask, labels
