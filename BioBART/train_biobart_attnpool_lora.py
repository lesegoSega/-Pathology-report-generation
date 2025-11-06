#!/usr/bin/env python3
import os, json, math, random, argparse, pathlib, gc, csv, time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

# ----------------------- utils -----------------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def load_reports_jsonl(path: str) -> Dict[str, str]:
    d={}
    with open(path, "r", encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            s=line.strip()
            if not s: continue
            try:
                o=json.loads(s)
            except Exception as e:
                print(f"[WARN] bad JSONL line {ln}: {e}")
                continue
            sid=o.get("slide_id") or o.get("id")
            rpt=o.get("report") or o.get("text") or o.get("target")
            if sid and isinstance(rpt,str) and rpt.strip():
                d[sid]=rpt.strip()
    return d  # :contentReference[oaicite:4]{index=4}

def parse_splits(p: str) -> Dict[str, List[str]]:
    with open(p,"r",encoding="utf-8") as f:
        return json.load(f)  # expects keys "train" and "val" :contentReference[oaicite:5]{index=5}

def load_patches(np_path: Path) -> np.ndarray:
    # robust NPZ / NPY loader (2D patch matrix fallback)  :contentReference[oaicite:6]{index=6}
    if np_path.suffix.lower()==".npy":
        arr=np.load(np_path,allow_pickle=False)
        if arr.ndim==1: return arr[None,:].astype(np.float32)
        if arr.ndim==2: return arr.astype(np.float32)
        return arr.reshape(arr.shape[0],-1).astype(np.float32)
    z=np.load(np_path,allow_pickle=False)
    for k in ["features","patches","feats","x","X","arr_0"]:
        if k in z:
            a=z[k]
            if a.ndim==1: return a[None,:].astype(np.float32)
            if a.ndim==2: return a.astype(np.float32)
            return a.reshape(a.shape[0],-1).astype(np.float32)
    best=None
    for k in z.files:
        a=z[k]
        if a.ndim==2 and (best is None or a.size>best.size): best=a
    if best is not None: return best.astype(np.float32)
    for k in z.files:
        a=z[k]
        if a.ndim==1: return a[None,:].astype(np.float32)
    raise ValueError(f"No 2D patches in {np_path.name}")

def maybe_find_feature(fid: str, feat_dir: Path) -> Optional[Path]:
    # try exact and common variants
    for ext in (".npz",".npy"):
        p = feat_dir / f"{fid}{ext}"
        if p.exists(): return p
    # permissive glob if exact not found
    for ext in (".npz",".npy"):
        hits = sorted(feat_dir.glob(f"{fid}*{ext}"))
        if hits: return hits[0]
    return None

def load_organ_map(csv_path:str)->Dict[str,Dict[str,str]]:
    if not csv_path: return {}
    m={}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid=r.get("slide_id") or r.get("id")
            if not sid: continue
            m[sid]={"organ":(r.get("organ") or "").strip(),
                    "site_detail":(r.get("site_detail") or "").strip()}
    return m  # :contentReference[oaicite:7]{index=7}

# ----------------- attention pooling -----------------
class AttnPool(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256):
        super().__init__()
        self.q = nn.Linear(in_dim, hidden, bias=False)
        self.k = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(in_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, in_dim, bias=False)

    def forward(self, x):  # x: [N, D]
        if x.ndim==3:
            N,T,D = x.shape
            x = x.view(N*T, D)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        att = torch.softmax((q @ k.T) / math.sqrt(q.size(-1)), dim=-1)
        pooled = att @ v
        pooled = self.out(pooled)
        # mean over tokens
        return pooled.mean(dim=0, keepdim=True)  # [1, D]

# --------------- dataset & collate -------------------
class NPZReportDataset(Dataset):
    def __init__(self, ids: List[str], feat_dir: str, reports: Dict[str,str], tokenizer, max_in_len:int):
        self.ids = ids
        self.feat_dir = Path(feat_dir)
        self.reports = reports
        self.tok = tokenizer
        self.max_in_len = max_in_len

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rep = self.reports.get(sid,"").strip()
        if not rep:
            raise KeyError(f"No report for {sid}")
        fp = maybe_find_feature(sid, self.feat_dir)
        if fp is None:
            raise FileNotFoundError(f"Feature not found for {sid}")
        patches = load_patches(fp)   # [T, D]
        return {"sid": sid, "patches": patches, "report": rep}

def collate_fn(batch, tok, max_target_len:int, attnpool:AttnPool, device):
    # pool to a single embedding vector per slide, then verbalize as a short prompt token sequence
    vecs = []
    for b in batch:
        x = torch.from_numpy(b["patches"]).to(device)
        with torch.no_grad():
            pooled = attnpool(x)  # [1, D]
        vecs.append(pooled.squeeze(0).cpu().numpy())
    V = np.stack(vecs, axis=0)  # [B, D]

    # turn pooled vectors into short pseudo-text tokens (bucketize) — lightweight, avoids huge inputs
    # (You can replace with learned projection+tokens later.)
    pseudo_texts = []
    for row in V:
        # coarse quantization to stable tokens
        ix = np.clip((row / (1e-6 + row.std() + 1e-6)).round().astype(int), -3, 3)
        pseudo = " ".join([f"f{j}:{ix[j]}" for j in range(0, min(64, ix.shape[0]), max(1, ix.shape[0]//64 or 1))])
        pseudo_texts.append("Features: " + pseudo)

    inputs = tok(pseudo_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")  # compact
    targets = tok([b["report"] for b in batch], padding=True, truncation=True,
                  max_length=max_target_len, return_tensors="pt")
    return {"input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]}

# -------------------- main train ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="GanjinZero/biobart-v2-base")
    ap.add_argument("--feat_dir", required=True, help="Directory with NPZ/NPY features")
    ap.add_argument("--reports_jsonl", required=True, help="reports.jsonl with {slide_id, report}")
    ap.add_argument("--splits_json", required=True, help="splits.json containing 'train' and 'val'")
    ap.add_argument("--slide_to_organ_csv", default="", help="optional for logging")
    ap.add_argument("--organ_terms_csv", default="", help="optional (not required)")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_tgt_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    splits = parse_splits(args.splits_json)   # uses "train"/"val"  :contentReference[oaicite:8]{index=8}
    reports = load_reports_jsonl(args.reports_jsonl)  # :contentReference[oaicite:9]{index=9}

    train_ids = [sid for sid in splits.get("train", []) if sid in reports]
    val_ids   = [sid for sid in splits.get("val",   []) if sid in reports]

    print(f"[DATA] train={len(train_ids)}  val={len(val_ids)}  (reports available: {len(reports)})")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # tokenizer & base model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # attach LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q","v","k","o","wi","wo"]  # broad coverage on BART
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # attention pool (feature_dim inferred lazily)
    # Peek one sample to infer D
    probe = maybe_find_feature(train_ids[0], Path(args.feat_dir))
    if probe is None:
        raise FileNotFoundError(f"No feature file for first training id: {train_ids[0]}")
    D = load_patches(probe).shape[1]
    attnpool = AttnPool(in_dim=D, hidden=min(512, max(128, D//4))).to(device)
    for p in attnpool.parameters():
        p.requires_grad_(False)  # keep it frozen for now; set True to learn pooling

    # datasets
    ds_tr = NPZReportDataset(train_ids, args.feat_dir, reports, tok, max_in_len=256)
    ds_va = NPZReportDataset(val_ids,   args.feat_dir, reports, tok, max_in_len=256)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True,
                       collate_fn=lambda b: collate_fn(b, tok, args.max_tgt_len, attnpool, device))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       collate_fn=lambda b: collate_fn(b, tok, args.max_tgt_len, attnpool, device))

    # optim
    steps_per_epoch = max(1, math.ceil(len(ds_tr)/args.batch_size))
    t_total = steps_per_epoch * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay":0.01},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio*t_total), t_total)

    best_val = 1e9
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss=0.0
        for step,b in enumerate(dl_tr,1):
            b = {k:v.to(device) for k,v in b.items()}
            out = model(**b)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            tr_loss += loss.item()
            if step % 50 == 0:
                print(f"[ep{ep}] step {step}/{steps_per_epoch} loss={tr_loss/step:.4f}")

        # val
        model.eval()
        va_loss=0.0; n=0
        with torch.no_grad():
            for b in dl_va:
                b = {k:v.to(device) for k,v in b.items()}
                out = model(**b)
                va_loss += out.loss.item(); n += 1
        va_loss = va_loss/max(1,n)
        print(f"[ep{ep}] train_loss={tr_loss/max(1,steps_per_epoch):.4f} | val_loss={va_loss:.4f}")

        # checkpoint best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "tok_name": args.model_name},
                       os.path.join(args.out_dir, "best.ckpt"))
            print(f"[SAVE] best.ckpt  val_loss={best_val:.4f}")

    # final save
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[DONE] saved to {args.out_dir}")

if __name__ == "__main__":
    main()
