"""
Stage 2B — Vision–Language Report Generation Training

This script trains the multimodal model to generate dermatopathology
microscopy reports from whole slide images.

Key features:
- Vision scale curriculum (stabilizes early training)
- Vision token dropout (improves robustness)
- Anchor loss (better report openings)
- Late unfreezing of Perceiver layers
- Vision token compression (K')
- LoRA-based training with frozen LLM

Outputs:
- Trained LoRA adapters
- VisionCompressor + Projector weights

Author: Solmaz Haddady
"""

import os, math, json, h5py, argparse, numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from peft import LoraConfig, TaskType, get_peft_model


from models.perceiver_projector import PositionalEncoderMLP, PerceiverResampler, Projector
from models.vision_compressor import VisionCompressor, VisionAuxHead


# -------------------- utils --------------------

def smart_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def cosine_with_warmup(opt, warmup, total, cycles=1):  ## controls LR  for OPT, gradualy reduces lr
    return get_cosine_with_hard_restarts_schedule_with_warmup(opt, warmup, total, cycles)

def count_trainable_params(module): 
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def find_subseq(hay, needle):
    n = len(needle)
    for s in range(0, len(hay)-n+1):
        if hay[s:s+n] == needle:
            return s
    return -1


# -------------------- dataset --------------------

class DermMicroscopyDataset(torch.utils.data.Dataset):
    """
    Returns:
      - feats:    [N, 768] patch features
      - xy:       [N, 2]   downsampled coordinates (x,y)
      - ids_kept: [Lt]     BOS + prefix-without-vision + target
      - insert_idx: int    where vision tokens are inserted
      - mask_kept: [Lt]bool 1 on RESPONSE content (excludes close tag), else 0
      - id, fd, cd: strings for logging
    """
    def __init__(self, csv_path, split_name, cfg):
        import pandas as pd
        self.cfg = cfg
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split_name].reset_index(drop=True)

        self.tags = cfg["prompt"]["tags"]
        self.instr = cfg["prompt"]["instruction"]
        self.max_len = int(cfg["text"]["max_len"])
        self.min_resp = int(cfg["text"].get("min_resp_tokens", 32))  # was 16; 32 works better for Stage-2
        self.feat_key = cfg["vision"]["features_key"]
        self.coord_key = cfg["vision"]["coords_key"]

        self.tok = AutoTokenizer.from_pretrained(cfg["text"]["tokenizer_name_or_path"], use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.bos_id = self.tok.bos_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        h5_path = r["features_h5"]
        with h5py.File(h5_path, "r") as h5:
            feats = torch.from_numpy(h5[self.feat_key][()]).float()
            coords= torch.from_numpy(h5[self.coord_key][()]).float()
        xy = coords[:,1:3]  # use xy only

        T = self.tags
        instr = self.instr
        max_len = int(self.max_len)
        bos_id = self.bos_id

        # 1) Build clean prefix & target strings
        prefix_str = (
            f"{instr}\n"
            f"{T['final_dx_open']}{r['FINAL_Diagnosis']}{T['final_dx_close']}\n"
            f"{T['critical_dx_open']}{r['CRITICAL_Diagnosis']}{T['critical_dx_close']}\n"
            f"{T['site_open']}unknown{T['site_close']}\n"               # for future  if any other infos can be extracted from  original report , in this file we do not have
            f"{T['context_open']}none{T['context_close']}\n"
            f"{T['vision_tag']}\n"
            f"{T['response_open']}"
        )
        target_text = (r["REPORT"] or "").strip()
        target_str = target_text + T["response_close"]

        # 2) Tokenize separately
        pid = self.tok(prefix_str, add_special_tokens=False)["input_ids"]
        tid = self.tok(target_str, add_special_tokens=False)["input_ids"]
        vision_ids = self.tok(T["vision_tag"], add_special_tokens=False)["input_ids"]
        close_ids  = self.tok(T["response_close"], add_special_tokens=False)["input_ids"]

        if len(vision_ids) == 0:
            raise ValueError("VISION tag tokenized to length 0; check tag string and tokenizer.")
        if len(close_ids) == 0:
            raise ValueError("response_close tokenized to length 0; check tag string and tokenizer.")

        # 3) Find the vision tag subsequence in the prefix tokens (insert fallback if missing)
        vpos = find_subseq(pid, vision_ids)
        if vpos < 0:
            vpos = max(0, len(pid)//2)
            pid = pid[:vpos] + vision_ids + pid[vpos:]

        # 4) Length budgeting with guarantees:
        #    - keep VISION region
        #    - keep at least min_resp content tokens + the close tag
        def budget_tokens(pid, vpos, vision_ids, tid, max_len, min_resp, close_ids):
            lhs = pid[:vpos]
            tag = pid[vpos:vpos+len(vision_ids)]
            rhs = pid[vpos+len(vision_ids):]
            keep_lhs, keep_rhs, keep_tid = lhs, rhs, tid

            def total_len():
                return len(keep_lhs) + len(tag) + len(keep_rhs) + len(keep_tid)

            # trim LHS (front) first
            while total_len() > max_len and len(keep_lhs) > 0:
                keep_lhs = keep_lhs[1:]
            # then trim RHS (end)
            while total_len() > max_len and len(keep_rhs) > 0:
                keep_rhs = keep_rhs[:-1]
            # finally trim TARGET (end) but keep min_resp + close_ids
            min_tid = min_resp + len(close_ids)
            while total_len() > max_len and len(keep_tid) > min_tid:
                keep_tid = keep_tid[:-1]
            return keep_lhs + tag + keep_rhs, keep_tid

        pid_b, tid_b = budget_tokens(pid, vpos, vision_ids, tid, max_len, self.min_resp, close_ids)

        # 5) Remove the vision tag tokens from the kept prefix (we’ll insert vision embeddings there)
        vpos_b = find_subseq(pid_b, vision_ids)
        assert vpos_b >= 0, "VISION tag must exist after budgeting"
        left = pid_b[:vpos_b]
        right= pid_b[vpos_b+len(vision_ids):]
        insert_idx = len(left) + (1 if bos_id is not None else 0)  # +1 for BOS

        #  Final ids_kept: [BOS] + left + right + target
        ids_kept = left + right + tid_b
        if bos_id is not None:
            ids_kept = [bos_id] + ids_kept

        #  Build mask_kept: 1 for target content tokens (excluding the close tag), else 0
        num_close = len(close_ids)
        num_target_total = len(tid_b)
        num_resp_content = max(0, num_target_total - num_close)
        mask_kept = [0] * (len(ids_kept) - num_target_total) + [1] * num_resp_content + [0] * num_close

        return {
            "id": str(r["Image_Nr"]),
            "feats": feats, "xy": xy,
            "ids_kept": torch.tensor(ids_kept, dtype=torch.long),
            "insert_idx": int(insert_idx),
            "mask_kept": torch.tensor(mask_kept, dtype=torch.bool),
            "fd": str(r["FINAL_Diagnosis"]),
            "cd": str(r["CRITICAL_Diagnosis"]),
        }


# -------------------- builders --------------------

def build_lm(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["text"]["tokenizer_name_or_path"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Compute in FP16 (matches 4-bit lm_head)
    if bool(cfg["text"].get("use_4bit", True)):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["text"]["model_name_or_path"],
            quantization_config=bnb,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["text"]["model_name_or_path"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

    # LoRA adapters
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = model.to(device)

    return tok, model


def build_vision(cfg, device, lm_dtype):
    vis = cfg["vision"]
    pos = PositionalEncoderMLP(in_dim=2, hidden=128, out_dim=vis["feat_dim"])
    res = PerceiverResampler(
        in_dim=vis["feat_dim"], dim=vis["latent_dim"],
        num_latents=vis["num_latents"], num_layers=vis["num_layers"],
        num_heads=vis["num_heads"]
    )
    cmp = VisionCompressor(dim=vis["latent_dim"], k_prime=vis["k_prime"], num_heads=vis["num_heads"])

    proj = Projector(in_dim=vis["latent_dim"], out_dim=4096)
    proj = proj.to(device=device, dtype=torch.float32)  # keep projector in fp32 internally

    # Warm-start pos/res from Stage-1
    pos_sd = torch.load(cfg["paths"]["init_pos_encoder"], map_location="cpu")
    res_sd = torch.load(cfg["paths"]["init_resampler"], map_location="cpu")
    pos.load_state_dict(pos_sd, strict=True)
    res.load_state_dict(res_sd, strict=True)

    # keep vision in fp32
    pos = pos.to(device=device, dtype=torch.float32)
    res = res.to(device=device, dtype=torch.float32)
    cmp = cmp.to(device=device, dtype=torch.float32)

    # freeze pos/res initially
    for p in pos.parameters(): p.requires_grad = False
    for p in res.parameters(): p.requires_grad = False

    # learnable gate to scale vision influence
    gate = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    gate_module = torch.nn.ParameterList([gate])

    return pos, res, cmp, proj, gate_module


# -------------------- forward --------------------

def forward_batch(model, tok, batch, cfg, device, pos, res, cmp, proj, gate_list, aux_head=None):
    """
    Forward pass with:
      - fp32 vision path, bounded & sanitized
      - dynamic vision_scale (ramp)
      - optional vision token dropout
      - masked CE + optional anchor loss
    """
    mdtype = next(model.lm_head.parameters()).dtype

    # ----- dynamic vision scale (curriculum) -----
    step = cfg.get("_step", 0)
    total_steps = cfg.get("_total_steps", 1)
    vscale_base = float(cfg["vision"].get("vision_scale", 0.10))
    vscale_target = float(cfg["vision"].get("vision_scale_max", vscale_base))
    frac = min(1.0, step / max(1, int(0.5 * total_steps)))  # reach target by 50% of training
    vision_scale = vscale_base + (vscale_target - vscale_base) * frac

    # unpack batch
    feats      = batch["feats"].to(device, dtype=torch.float32)
    xy         = batch["xy"].to(device, dtype=torch.float32)
    ids_kept   = batch["ids_kept"].to(device).unsqueeze(0)  # [1,Lt]
    insert_idx = int(batch["insert_idx"])
    mask_kept  = batch["mask_kept"].to(device).unsqueeze(0) # [1,Lt] bool

    # ---------- VISION (fp32) ----------
    gate = gate_list[0] if (gate_list is not None and len(gate_list) > 0) else None
    with torch.no_grad():
        fused  = feats + pos(xy)                # [N,768] fp32
        fused  = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        media  = fused.unsqueeze(0)             # [1,N,768] fp32
        lat    = res(media)                     # [1,640,1536] fp32
        if isinstance(lat, (tuple, list)):
            lat = lat[0]
    zc = cmp(lat)                               # [1,K',1536] fp32
    zc = torch.nan_to_num(zc, nan=0.0, posinf=1e4, neginf=-1e4)

    # optional k_prime schedule (slice)
    kprime_now = int(cfg["vision"].get("k_prime", zc.size(1)))
    if zc.size(1) > kprime_now:
        zc = zc[:, :kprime_now, :]

    v32 = proj(zc)                              # [1,K',4096] fp32
    v32 = torch.tanh(v32)
    if gate is not None:
        v32 = v32 * (torch.sigmoid(gate) * 1.5 + 0.25)  # ~0.25..1.75x
    v32 = v32 * vision_scale
    v32 = torch.nan_to_num(v32, nan=0.0, posinf=5.0, neginf=-5.0)
    vtok = v32.to(mdtype)  # cast to LLM dtype only here

    # vision token dropout (robustness)
    if model.training:
        drop_prob = float(cfg["vision"].get("vision_token_dropout", 0.0))
        if drop_prob > 0:
            keep = (torch.rand(vtok.size()[:2], device=vtok.device) > drop_prob).unsqueeze(-1)  # [1,K',1]
            vtok = vtok * keep

    # telemetry 
    if (np.random.rand() < 0.01):
        m = float(vtok.abs().mean().item())
        mx = float(vtok.abs().max().item())
        print(f"[telemetry] vtok mean|max = {m:.4f} | {mx:.4f}", flush=True)
        print(f"[telemetry] compressor K'={zc.size(1)} | vtok shape={vtok.shape}")

    # ---------- TEXT (LLM dtype) ----------
    text_emb = model.get_input_embeddings()(ids_kept).to(mdtype)   # [1,Lt,4096]
    attn_txt = torch.ones_like(ids_kept)

    # splice vision tokens
    if not (0 <= insert_idx <= text_emb.size(1)):
        return None, {"seq_len": int(text_emb.size(1)), "skipped": True}

    before = text_emb[:, :insert_idx, :]
    after  = text_emb[:, insert_idx:, :]
    inputs = torch.cat([before, vtok, after], dim=1)               # [1,Lt+K',4096]
    attn   = torch.cat([
        attn_txt[:, :insert_idx],
        torch.ones((1, vtok.size(1)), device=device, dtype=attn_txt.dtype),
        attn_txt[:, insert_idx:]
    ], dim=1)

    # build labels (only on response span)
    labels = torch.full((1, inputs.size(1)), -100, dtype=torch.long, device=device)
    # left chunk
    L = insert_idx
    if L > 0:
        labels[:, :L] = ids_kept[:, :L]
        labels[:, :L] = torch.where(
            mask_kept[:, :L],
            labels[:, :L],
            torch.tensor(-100, device=device)
        )
    # right chunk
    R = ids_kept.size(1) - insert_idx
    if R > 0:
        labels[:, L + vtok.size(1): L + vtok.size(1) + R] = ids_kept[:, insert_idx:]
        labels[:, L + vtok.size(1): L + vtok.size(1) + R] = torch.where(
            mask_kept[:, insert_idx:],
            labels[:, L + vtok.size(1): L + vtok.size(1) + R],
            torch.tensor(-100, device=device)
        )

    # must have at least one supervised token
    if (labels != -100).sum().item() == 0:
        return None, {"seq_len": int(inputs.size(1)), "skipped": True}

    # ---------- forward with auto-backoff if needed ----------
    loss_final = None
    for backoff in [1.0, 0.5, 0.25]:
        try_inputs = inputs
        if backoff < 1.0:
            try_inputs = torch.cat([before, vtok * backoff, after], dim=1)
        out = model(inputs_embeds=try_inputs, attention_mask=attn, labels=labels)
        if torch.isfinite(out.loss):
            loss = out.loss

            # ----- optional anchor loss on the first tokens of response -----
            anchor_len = int(cfg["train"].get("anchor_len", 0))
            anchor_w   = float(cfg["train"].get("anchor_weight", 0.0))
            if anchor_len > 0 and anchor_w > 0:
                sup_positions = torch.nonzero(labels != -100, as_tuple=False)
                if sup_positions.numel() > 0:
                    start = int(sup_positions[0][1].item())
                    end   = min(start + anchor_len, labels.size(1))
                    labels_anchor = torch.full_like(labels, -100)
                    labels_anchor[:, start:end] = labels[:, start:end]
                    out_anchor = model(inputs_embeds=try_inputs, attention_mask=attn, labels=labels_anchor)
                    loss = (1.0 - anchor_w) * loss + anchor_w * out_anchor.loss

            loss_final = loss
            info = {"seq_len": int(inputs.size(1)), "skipped": False, "backoff": backoff}
            break

    if loss_final is None:
        return None, {"seq_len": int(inputs.size(1)), "skipped": True, "backoff": 0.0}

    return loss_final, info


# -------------------- train / eval / save --------------------

def evaluate(model, tok, dl, cfg, device, pos, res, cmp, proj, gate_list, aux_head=None):
    model.eval(); proj.eval(); cmp.eval()
    losses = []
    with torch.no_grad():
        for batch in dl:
            loss, _ = forward_batch(model, tok, batch, cfg, device, pos, res, cmp, proj, gate_list, aux_head)
            if (loss is None) or (not torch.isfinite(loss)):
                continue
            losses.append(float(loss.item()))
    model.train(); proj.train(); cmp.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_ckpt(model, proj, cmp, res_maybe, path: Path, cfg, val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "val_loss": float(val_loss) if np.isfinite(val_loss) else None,
        "lora": model.state_dict(),       # adapters only (PEFT)
        "projector": proj.state_dict(),
        "compressor": cmp.state_dict(),
        "config": cfg
    }
    if res_maybe is not None:
        sub = {}
        if hasattr(res_maybe, "layers"):
            for i, layer in enumerate(res_maybe.layers):
                for n, p in layer.named_parameters():
                    if p.requires_grad:
                        sub[f"layers.{i}.{n}"] = p.detach().cpu()
        ckpt["resampler_last"] = sub
    torch.save(ckpt, path)
    print(f"Saved → {str(path)}")


# -------------------- main --------------------

def unfreeze_perceiver_tail(resampler, n_tail=2):
    if not hasattr(resampler, "layers"):  # safety
        return
    L = len(resampler.layers)
    for i, layer in enumerate(resampler.layers):
        req = (i >= L - n_tail)
        for p in layer.parameters():
            p.requires_grad = req

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.save_dir is not None:
        cfg["paths"]["save_dir"] = args.save_dir

    # safety defaults
    cfg["train"]["eval_every"] = int(cfg["train"].get("eval_every", 200))
    cfg["train"]["save_every"] = int(cfg["train"].get("save_every", 100))

    device = smart_device()
    print("Device:", device, flush=True)

    # --- LLM + LoRA ---
    tok, model = build_lm(cfg, device)

    # Align vocab with stage-1
    st1 = torch.load(cfg["paths"]["stage1_best_ckpt"], map_location="cpu")
    st1_vocab = st1["lora"]["base_model.model.model.embed_tokens.weight"].shape[0]
    cur_vocab = model.get_input_embeddings().weight.shape[0]
    if cur_vocab != st1_vocab:
        print(f"Resizing token embeddings: cur={cur_vocab} -> st1={st1_vocab}", flush=True)
        model.resize_token_embeddings(st1_vocab, mean_resizing=True)

    # Load LoRA adapters
    missing, unexpected = model.load_state_dict(st1["lora"], strict=False)
    print(f"LoRA loaded. missing: {len(missing)} unexpected: {len(unexpected)}", flush=True)

    # temp projector just to verify SD
    tmp_proj = Projector(in_dim=cfg["vision"]["latent_dim"], out_dim=4096)
    tmp_proj.load_state_dict(st1["projector"], strict=True)
    del tmp_proj

    # --- Vision modules ---
    lm_dtype = next(model.lm_head.parameters()).dtype
    pos, res, cmp, proj, gate_list = build_vision(cfg, device, lm_dtype)

    # reload projector weights onto correct device/dtype
    proj.load_state_dict(st1["projector"], strict=True)
    print("Projector loaded from Stage-1.", flush=True)

    # --- Datasets ---
    ds_tr = DermMicroscopyDataset(cfg["paths"]["splits_csv"], cfg["train"]["splits"]["train"], cfg)
    ds_va = DermMicroscopyDataset(cfg["paths"]["splits_csv"], cfg["train"]["splits"]["val"], cfg)

    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,
                       num_workers=0, persistent_workers=False, pin_memory=False,
                       collate_fn=lambda b: b[0])
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=0, persistent_workers=False, pin_memory=False,
                       collate_fn=lambda b: b[0])

    print(f"Train samples: {len(ds_tr)} | Val samples: {len(ds_va)}", flush=True)
    print(">> Entering training loop", flush=True)

    # --- Trainables ---
    for p in pos.parameters(): p.requires_grad = False
    for p in res.parameters(): p.requires_grad = False
    for p in cmp.parameters(): p.requires_grad = True
    for p in proj.parameters(): p.requires_grad = True
    aux_head = None

    tr_lora = [p for p in model.parameters() if p.requires_grad]
    optim_groups = [
        {"params": tr_lora, "lr": float(cfg["train"]["lr_lora"])},
        {"params": [p for p in proj.parameters() if p.requires_grad], "lr": float(cfg["train"]["lr_proj"])},
        {"params": [p for p in cmp.parameters() if p.requires_grad], "lr": float(cfg["train"]["lr_compressor"])},
    ]
    # gate param (if present)
    if gate_list is not None and hasattr(gate_list, "parameters"):
        gate_params = list(gate_list.parameters())
        if gate_params:
            optim_groups.append({"params": gate_params, "lr": float(cfg["train"].get("lr_gate", 1e-4))})

    opt = torch.optim.AdamW(optim_groups, weight_decay=float(cfg["train"]["weight_decay"]))

    total_steps = cfg["train"]["epochs"] * math.ceil(len(ds_tr) / cfg["train"]["grad_accum"])
    sch = cosine_with_warmup(opt, int(cfg["train"]["warmup_steps"]), total_steps)

    save_dir = Path(cfg["paths"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    model.train(); proj.train(); cmp.train()

    # counts
    print(f"[counts] LoRA (PEFT): {count_trainable_params(model):,}", flush=True)
    print(f"[counts] Projector  : {count_trainable_params(proj):,}", flush=True)
    print(f"[counts] Compressor : {count_trainable_params(cmp):,}", flush=True)
    #print(f"[telemetry] vtok shape = {vtok.shape}")  # [1, K', 4096] => K' should be 256

    running = 0.0
    accum = 0

    # Late unfreeze setup
    late_unfired = False
    late_at = int(float(cfg["train"].get("late_unfreeze_at_frac", 0.8)) * total_steps)
    lr_perceiver_late = float(cfg["train"].get("lr_perceiver_late", 2e-5))

    try:
        for ep in range(cfg["train"]["epochs"]):
            for nb, batch in enumerate(dl_tr, start=1):
                print(f"[epoch {ep+1}] batch {nb} id={batch['id']}", flush=True)

                # pass scheduler counters for curriculum
                cfg["_step"] = step
                cfg["_total_steps"] = total_steps

                loss, info = forward_batch(model, tok, batch, cfg, device, pos, res, cmp, proj, gate_list, aux_head)
                if (loss is None) or (not torch.isfinite(loss)):
                    print(f"[epoch {ep+1}] batch {nb} -> skipped (non-finite loss)", flush=True)
                    continue

                (loss / cfg["train"]["grad_accum"]).backward()
                running += float(loss.item())
                accum += 1

                if (accum % cfg["train"]["grad_accum"]) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g["params"] if p.requires_grad],
                        max_norm=float(cfg["train"]["clip_grad_norm"])
                    )
                    opt.step(); sch.step(); opt.zero_grad()
                    step += 1

                    avg_loss = running / cfg["train"]["grad_accum"]
                    print(f"[epoch {ep+1}] step {step} | train loss (avg/{cfg['train']['grad_accum']}): {avg_loss:.4f}", flush=True)
                    running = 0.0

                    # Late unfreeze trigger (only once)
                    if (not late_unfired) and (step >= late_at):
                        unfreeze_perceiver_tail(res, n_tail=2)
                        # add new param group for newly unfrozen perceiver params
                        optim_groups.append({
                            "params": [p for p in res.parameters() if p.requires_grad],
                            "lr": lr_perceiver_late
                        })
                        opt = torch.optim.AdamW(optim_groups, weight_decay=float(cfg["train"]["weight_decay"]))
                        sch = cosine_with_warmup(opt, int(cfg["train"]["warmup_steps"]), total_steps)
                        late_unfired = True
                        print(f"[late-unfreeze] Enabled last 2 Perceiver layers @ step {step} with lr={lr_perceiver_late}", flush=True)

                    # periodic autosave
                    if (step % cfg["train"]["save_every"]) == 0:
                        save_ckpt(model, proj, cmp, None, save_dir / f"stage2_step{step}.pt", cfg, val_loss=float("nan"))

                    # periodic validation + best save
                    if (cfg["train"]["eval_every"] > 0) and ((step % cfg["train"]["eval_every"]) == 0):
                        try:
                            val_loss = evaluate(model, tok, dl_va, cfg, device, pos, res, cmp, proj, gate_list, aux_head=None)
                            print(f"  >> val loss: {val_loss:.4f}", flush=True)
                            if val_loss < best_val:
                                best_val = val_loss
                                save_ckpt(model, proj, cmp, None, save_dir / "stage2_best.pt", cfg, val_loss)
                                print(f"  >> saved best checkpoint @ val_loss={val_loss:.4f}", flush=True)
                        except Exception as e:
                            print(f"[warn] eval failed: {e}; continuing training.", flush=True)

            # end-of-epoch eval + save
            try:
                val_loss = evaluate(model, tok, dl_va, cfg, device, pos, res, cmp, proj, gate_list, aux_head=None)
                print(f"[epoch {ep+1}] end val loss: {val_loss:.4f}", flush=True)
            except Exception as e:
                val_loss = float("inf")
                print(f"[warn] end-of-epoch eval failed: {e}", flush=True)

            save_ckpt(model, proj, cmp, None, save_dir / f"stage2_e{ep+1}.pt", cfg, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(model, proj, cmp, None, save_dir / "stage2_best.pt", cfg, val_loss)
                print(f"  >> new best after epoch {ep+1}: {val_loss:.4f}", flush=True)

    finally:
        # always save a "last" checkpoint on exit
        save_ckpt(model, proj, cmp, None, save_dir / "stage2_last.pt", cfg, val_loss=float("nan"))
        print("[final] Saved 'stage2_last.pt' (no-val).", flush=True)


if __name__ == "__main__":
    main()
