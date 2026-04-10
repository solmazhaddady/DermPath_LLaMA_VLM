 
"""
Stage 2-A: Vision–Language Alignment Training

This stage extends the slide-level visual encoder into a multimodal
vision–language model (VLM). The goal is not to generate perfect reports
yet, but to ensure that visual features and language representations are
properly aligned.

Only the projector and LoRA adapters are trained, while the vision encoder
and language model backbone remain frozen.

Author: Solmaz Haddady

"""

import os, json, math, time, random, argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_with_hard_restarts_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Any, Tuple, List

# ---------- Import  modules ----------
# 
from models.vlm_projector import PositionalEncoderMLP, PerceiverResampler, Projector


# ----------------- Utils -----------------
def set_seed(seed:int=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def smart_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

# Mask the response span only
def mask_for_response(tokenizer, text:str, open_tag:str, close_tag:str, max_len:int, add_bos:bool):
    enc = tokenizer(
        text, max_length=max_len, truncation=True, padding=False,
        add_special_tokens=False, return_offsets_mapping=True
    )
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    s = text.find(open_tag); e = text.rfind(close_tag)
    assert s >= 0 and e > s, "Response tags missing or misordered."
    s += len(open_tag)
    mask = []
    if add_bos: mask.append(0)
    for (a,b) in offs:
        inter = max(0, min(b, e) - max(a, s))
        mask.append(1 if inter > 0 else 0)
    return ids, offs, mask

def build_prompt(row:Dict[str,Any], cfg:Dict[str,Any]) -> Tuple[str, str]:
    T = cfg["prompt"]["tags"]
    instr = cfg["prompt"]["instruction"]
    fd = row["FINAL_Diagnosis"]
    cd = row["CRITICAL_Diagnosis"]
    prefix = (
        f"{instr}\n"
        f"{T['final_dx_open']}{fd}{T['final_dx_close']}\n"
        f"{T['critical_dx_open']}{cd}{T['critical_dx_close']}\n"
        f"{T['site_open']}unknown{T['site_close']}\n"
        f"{T['context_open']}none{T['context_close']}\n"
        f"{T['vision_tag']}\n"
        f"{T['response_open']}"
    )
    target = (row["REPORT"] or "").strip()
    return prefix + target + T["response_close"], target


# ----------------- Dataset -----------------
class DermMicroscopyDataset(Dataset):
    def __init__(self, csv_path:str, split:str, cfg:Dict[str,Any]):
        self.cfg = cfg
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.tok = AutoTokenizer.from_pretrained(cfg["text"]["tokenizer_name_or_path"], use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_len = int(cfg["text"]["max_len"])
        self.T = cfg["prompt"]["tags"]
        self.fkey = cfg["vision"]["features_key"]
        self.ckey = cfg["vision"]["coords_key"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx:int):
        r = self.df.iloc[idx]
        full_text, _ = build_prompt(r, self.cfg)

        # tokenize + offsets
        ids, offs, resp_mask = mask_for_response(self.tok, full_text, self.T["response_open"], self.T["response_close"], self.max_len, add_bos=True)
        vtag_pos = full_text.find(self.T["vision_tag"])
        vtag_end = vtag_pos + len(self.T["vision_tag"])
        assert vtag_pos >= 0

        # find where to insert vision tokens by removing the <VISION> tag tokens
        keep = []; insert_idx = None
        # offs excludes BOS; we added BOS logically, so we’ll account later
        for i,(a,b) in enumerate(offs):
            inter = max(0, min(b, vtag_end) - max(a, vtag_pos))
            if inter > 0:
                if insert_idx is None:
                    insert_idx = len(keep)
                continue
            keep.append(i)
        if insert_idx is None:
            insert_idx = len(keep)//2

        # Apply keep and add BOS
        ids_kept = [self.tok.bos_token_id] + [ids[i] for i in keep]
        resp_mask_kept = [0] + [resp_mask[i+1] for i in keep] if len(resp_mask)>0 else [0]*(len(ids_kept))

        return {
            "row": r.to_dict(),
            "ids_kept": torch.tensor(ids_kept, dtype=torch.long),
            "resp_mask_kept": torch.tensor(resp_mask_kept, dtype=torch.bool),
            "insert_idx": insert_idx + 1,  # +1 for BOS
        }


def collate_pad(batch, pad_id:int):
    B = len(batch)
    maxL = max(len(x["ids_kept"]) for x in batch)
    input_ids = torch.full((B,maxL), pad_id, dtype=torch.long)
    resp_mask = torch.zeros((B,maxL), dtype=torch.bool)
    insert_idx = []
    rows = []
    for i,ex in enumerate(batch):
        L = len(ex["ids_kept"])
        input_ids[i,:L] = ex["ids_kept"]
        resp_mask[i,:L] = ex["resp_mask_kept"]
        insert_idx.append(ex["insert_idx"])
        rows.append(ex["row"])
    return {
        "input_ids": input_ids,
        "resp_mask": resp_mask,
        "insert_idx": torch.tensor(insert_idx, dtype=torch.long),
        "rows": rows
    }


# -------------- Vision modules --------------  frozen 
def build_frozen_vision(cfg, device, model_dtype):
    pos = PositionalEncoderMLP(in_dim=2, hidden=128, out_dim=768)
    resampler = PerceiverResampler(in_dim=768, dim=1536, num_latents=640, num_layers=6, num_heads=16)   
    pos_sd = torch.load(cfg["paths"]["init_pos_encoder"], map_location="cpu")
    resampler_sd = torch.load(cfg["paths"]["init_resampler"], map_location="cpu")
    pos.load_state_dict(pos_sd, strict=True)
    resampler.load_state_dict(resampler_sd, strict=True)
    for p in pos.parameters(): p.requires_grad = False
    for p in resampler.parameters(): p.requires_grad = False
    pos = pos.to(device)
    resampler = resampler.to(device)
    # projector is trainable and should output LLM hidden size
    projector = Projector(in_dim=1536, out_dim=4096).to(device)
    # Make sure its parameters are in model dtype
    #proj = proj.to(dtype=model_dtype)
    #return pos, res, proj
    # return pos, res, proj
    return pos, resampler, projector



# -------------- Build LLM + LoRA --------------
def build_lm(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["text"]["tokenizer_name_or_path"], use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # switch to float16 if needed
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["lm"]["model_name_or_path"],
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    try: model.gradient_checkpointing_enable()
    except: pass

    # ensure embeddings match tokenizer
    if len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    # LoRA
    lcfg = cfg["lm"]["lora"]
    peft_cfg = LoraConfig(
        r=lcfg["rank"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return tok, model



# -------------- Forward (one batch) --------------
def forward_batch(model, tok, batch, cfg, device, pos, resampler, projector):
    model_dtype = next(model.parameters()).dtype

    input_ids = batch["input_ids"].to(device)              # [B, Lt]
    text_emb  = model.get_input_embeddings()(input_ids)    # [B, Lt, H]
    text_emb  = text_emb.to(model_dtype)
    attn_text = (input_ids != tok.pad_token_id).long()     # [B, Lt]
    resp_mask = batch["resp_mask"].to(device)              # [B, Lt] bool

    B, Lt = input_ids.shape
    new_inputs, new_attn, new_labels = [], [], []
    supervised_counts = []

    for i in range(B):
        row = batch["rows"][i]
        insert_idx = int(batch["insert_idx"][i].item())
        insert_idx = max(0, min(insert_idx, Lt))

        # --- load vision features ---
        with h5py.File(row["features_h5"], "r") as h5:
            feats = torch.from_numpy(h5[cfg["vision"]["features_key"]][()]).float().to(device)   # [N,768]
            coords= torch.from_numpy(h5[cfg["vision"]["coords_key"]][()]).float().to(device)     # [N,3]
        xy = coords[:,1:3]

        # --- vision forward (float32), then projector (float32→model_dtype out) ---
        with torch.no_grad():
            fused   = feats + pos(xy)               # [N,768], float32
            media   = fused.unsqueeze(0)            # [1,N,768]
            latents = resampler(media)                    # [1,640,1536], float32

        # Project (internally float32, output cast to projector param dtype)
        vtokens = projector(latents)                     # [1,640,H_out]; H_out dtype = projector param dtype
        # Ensure final dtype matches LLM hidden (half/bf16)
        vtokens = vtokens.to(model_dtype)
        K = vtokens.size(1)

        # --- splice ---
        te = text_emb[i:i+1]                        # [1, Lt, H]
        before = te[:, :insert_idx, :]  # insert positioon 
        after  = te[:, insert_idx:, :]
        inp = torch.cat([before, vtokens, after], dim=1)  # [1, Lt+K, H]
        new_inputs.append(inp)

        # --- attention ---
        a = attn_text[i:i+1]                        # [1, Lt]
        a_new = torch.cat(
            [a[:, :insert_idx],
             torch.ones((1, K), dtype=a.dtype, device=device),
             a[:, insert_idx:]],
            dim=1
        )                                           # [1, Lt+K]
        new_attn.append(a_new)

        # --- labels (only response span) ---
        kept_ids = input_ids[i]                     # [Lt]
        total_len = inp.size(1)                     # Lt + K
        lab = torch.full((1, total_len), -100, dtype=torch.long, device=device)

        # left side
        left_len = min(insert_idx, Lt)
        if left_len > 0:
            lab[:, :left_len] = kept_ids[:left_len]

        # right side (only as many as fit)
        right_src_len = Lt - insert_idx
        right_dst_len = total_len - (insert_idx + K)
        n_right = max(0, min(right_src_len, right_dst_len))
        if n_right > 0:
            lab[:, insert_idx + K : insert_idx + K + n_right] = kept_ids[insert_idx : insert_idx + n_right]

        # apply response mask
        m = resp_mask[i:i+1]                        # [1, Lt]
        if left_len > 0:
            lab[:, :left_len] = torch.where(
                m[:, :left_len], lab[:, :left_len], torch.tensor(-100, device=device)
            )
        if n_right > 0:
            lab[:, insert_idx + K : insert_idx + K + n_right] = torch.where(
                m[:, insert_idx : insert_idx + n_right],
                lab[:, insert_idx + K : insert_idx + K + n_right],
                torch.tensor(-100, device=device)
            )

        sup_count = int((lab != -100).sum().item())
        supervised_counts.append(sup_count)
        new_labels.append(lab)

    inputs = torch.cat(new_inputs, dim=0)  # [B, Lt+K, H]
    attn   = torch.cat(new_attn,   dim=0)  # [B, Lt+K]
    labels = torch.cat(new_labels, dim=0)  # [B, Lt+K]

    # if no supervised tokens in the whole batch, skip (return None)
    if sum(supervised_counts) == 0:
        return None, None

    out = model(inputs_embeds=inputs, attention_mask=attn, labels=labels)  #####

    # guard against NaN/Inf loss
    if not torch.isfinite(out.loss):
        print(f"[warn] non-finite loss detected (loss={out.loss}). Skipping batch.")
        return None, None

    return out.loss, out

# -------------- Training loop --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split_csv", type=str, default=None)  # override if needed
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--save_dir", type=str, default="./checkpoints_stage1")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_json(args.config)
    if args.split_csv is None:
        split_csv = cfg["paths"]["splits_csv"]
    else:
        split_csv = args.split_csv

    device = smart_device()
    print("Device:", device)

    # Build LLM + LoRA
    tok, model = build_lm(cfg, device)
    model = model.to(device)
    model_dtype = next(model.parameters()).dtype

    # Vision modules
    pos, resampler, projector = build_frozen_vision(cfg, device, model_dtype) #####=================

    # Datasets
    ds_tr = DermMicroscopyDataset(split_csv, cfg["train"]["splits"]["train"], cfg)
    ds_va = DermMicroscopyDataset(split_csv, cfg["train"]["splits"]["val"], cfg)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda b: collate_pad(b, tok.pad_token_id))

    # Optimizer: LoRA + projector only
    trainable = list(model.parameters())  # LoRA params are flagged requires_grad=True
    projector_params = [p for p in projector.parameters() if p.requires_grad]
    all_params = [
        {"params": [p for p in trainable if p.requires_grad], "lr": args.lr},
        {"params": projector_params, "lr": args.lr},
    ]
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    # Scheduler (cosine with restarts; optional)
    total_steps = args.epochs * math.ceil(len(ds_tr) / (args.batch_size * args.grad_accum_steps))
    sch = get_cosine_with_hard_restarts_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps, num_cycles=1
    )

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    step = 0; best_val = float("inf")
    
    model.train(); projector.train()
    for epoch in range(args.epochs):
        running = 0.0
        for nb, batch in enumerate(dl_tr, start=1):
            fb = forward_batch(model, tok, batch, cfg, device, pos, resampler, projector)
            if fb is None or fb[0] is None:
            # nothing to learn from this batch or non-finite loss
                continue

            loss, _ = fb
            (loss / args.grad_accum_steps).backward()
            running += float(loss.item())

            if nb % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + projector_params,
                    max_norm=1.0
                )
                opt.step(); sch.step(); opt.zero_grad()
                step += 1

                if step % 10 == 0:
                    print(f"[epoch {epoch+1}] step {step} | train loss (avg/{args.grad_accum_steps}): {running/args.grad_accum_steps:.4f}")
                running = 0.0

                if step % args.eval_every == 0:
                    val_loss = evaluate(model, tok, dl_va, cfg, device, pos, resampler, projector)
                    print(f"  >> val loss: {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(model, projector, save_dir / f"stage1_best.pt", cfg, val_loss)
                        print(f"  >> saved best checkpoint @ val_loss={val_loss:.4f}")

# end epoch eval + save last
    val_loss = evaluate(model, tok, dl_va, cfg, device, pos, resampler, projector)
    save_ckpt(model, projector, save_dir / f"stage1_e{epoch+1}.pt", cfg, val_loss)
    if val_loss < best_val:
        best_val = val_loss
        save_ckpt(model, projector, save_dir / f"stage1_best.pt", cfg, val_loss)
    print(f"Epoch {epoch+1} done. val_loss={val_loss:.4f}")
    
def evaluate(model, tok, dl, cfg, device, pos, resampler, projector):
    model.eval(); projector.eval()
    losses = []
    with torch.no_grad():
        for batch in dl:
            fb = forward_batch(model, tok, batch, cfg, device, pos, resampler, projector)
            if fb is None or fb[0] is None:
                continue
            loss, _ = fb
            losses.append(loss.item())
    model.train(); projector.train()
    return float(np.mean(losses)) if losses else float("inf")

def save_ckpt(model, projector, path:Path, cfg, val_loss:float):
    ckpt = {
        "val_loss": val_loss,
        "lora": model.state_dict(),      # PEFT adapters
        "projector": projector.state_dict(),
        "config": cfg,
    }
    torch.save(ckpt, path)
    print(f"Saved → {str(path)}")



if __name__ == "__main__":
    main()
