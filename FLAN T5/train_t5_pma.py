#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAN-T5-base + PMA (multi-head attention pooling over patch features) + LoRA,
for slide-to-report generation.

Key points:
- Loads PATCH-LEVEL features from .npz (N x D, often D=1024).
- PMA produces K visual tokens (K<<N) as a compact "visual prefix" for the encoder.
- Keeps organ/site hinting in the prompt (lexicon or CSV map).
- Long sequence training/decoding defaults so the model learns long reports.
- Metrics: ROUGE-L, BERTScore (rescaled), BLEU-4, clinical term recall.
- AMP mixed precision for speed.

Install (once):
  pip install -U sentencepiece "transformers>=4.40" peft evaluate rouge-score bert-score sacrebleu pandas
"""

import math
import json
import random
import argparse
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_cosine_schedule_with_warmup,
)

import evaluate
from peft import LoraConfig, get_peft_model
import pandas as pd


# -------------------- Organ hinting --------------------
_ORGAN_LEXICON = {
    "liver": ["liver", "hepatic", "hepatocellular", "bile duct", "cholangiocarcinoma"],
    "gallbladder": ["gallbladder", "cholecyst", "cholecystitis", "cholecystectomy"],
    "pancreas": ["pancreas", "pancreatic", "ampulla", "ampullary"],
    "colon": ["colon", "colonic", "colorectal", "cecum", "ascending colon", "sigmoid"],
    "stomach": ["stomach", "gastric"],
    "lung": ["lung", "pulmonary", "bronch", "peribronchial"],
    "breast": ["breast", "mammary"],
    "kidney": ["kidney", "renal"],
    "prostate": ["prostate", "prostatic"],
    "brain": ["brain", "glioblastoma", "astrocytoma"],
    "skin": ["skin", "dermal", "cutaneous"],
    "ovary": ["ovary", "ovarian", "adnexa"],
    "uterus": ["uterus", "endometrium", "endometrial", "myometrium", "cervix", "cervical"],
    "bladder": ["bladder", "urothelial"],
    "thyroid": ["thyroid"],
    "esophagus": ["esophagus", "esophageal"],
    "small intestine": ["duodenum", "jejunum", "ileum", "small intestine", "small bowel"],
}

_TERMS_OF_INTEREST = [
    "diagnosis", "margin", "margins", "invasion", "venous invasion",
    "stage", "ajcc", "pt", "pn", "pm", "glypican-3", "immunohistochemistry",
    "well-differentiated", "moderately differentiated", "poorly differentiated",
    "steatosis", "necrosis", "fibrosis", "perineural", "lymphovascular"
]

def _extract_organ_from_text(txt: str) -> Optional[str]:
    low = txt.lower()
    best, hits = None, 0
    for organ, keys in _ORGAN_LEXICON.items():
        h = sum(1 for k in keys if k in low)
        if h > hits:
            best, hits = organ, h
    return best


# -------------------- PMA pooling --------------------
class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention (Set Transformer style):
      Inputs: X in R^{B x N x Din}
      Outputs: K tokens in R^{B x K x dm}
    """
    def __init__(self, in_dim=1024, dm=768, K=12, nheads=8, dropout=0.1):
        super().__init__()
        self.K = K
        self.proj_in = nn.Linear(in_dim, dm)
        self.queries = nn.Parameter(torch.randn(K, dm))
        self.attn = nn.MultiheadAttention(dm, nheads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dm)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):  # X: (B, N, Din)
        V = self.proj_in(X)                                  # (B, N, dm)
        Q = self.queries.unsqueeze(0).expand(X.size(0), -1, -1)  # (B, K, dm)
        Y, _ = self.attn(Q, V, V)                            # (B, K, dm)
        Y = self.ln(Y)
        Y = self.drop(Y)
        return Y


# -------------------- Data utilities --------------------
def load_patch_matrix(npz_path: str) -> np.ndarray:
    """
    Load (N,D) patch feature matrix from .npz.
    Strategy:
      - Prefer any 2D array with D=1024
      - Else pick the largest (N*D) 2D array
      - Else if only 1D, treat as single-vector (N=1)
    """
    with np.load(npz_path) as z:
        best = None
        best_size = -1
        for k in z.files:
            arr = z[k]
            if arr.ndim == 2:
                if arr.shape[1] == 1024:
                    return arr.astype("float32")
                size = arr.shape[0] * arr.shape[1]
                if size > best_size:
                    best = arr; best_size = size
        if best is not None:
            return best.astype("float32")
        # Fallback: any array
        arr0 = z[z.files[0]].astype("float32")
        if arr0.ndim == 1:
            return arr0[None, :]  # (1, D)
        return arr0


def subsample_patches(X: np.ndarray, max_patches: int, mode: str = "uniform") -> np.ndarray:
    """
    Subsample to at most max_patches along N, keeping distribution simple.
    mode: 'uniform' or 'first'
    """
    N = X.shape[0]
    if N <= max_patches:
        return X
    if mode == "first":
        return X[:max_patches]
    # uniform without replacement
    idx = np.linspace(0, N - 1, num=max_patches, dtype=int)
    return X[idx]


# -------------------- Dataset --------------------
class SlideReportPMADataset(Dataset):
    """
    Uses patch features: "<slide_id>.npz" with (N, D) patches inside.
    Builds a structured prompt + optional organ hint.
    """
    def __init__(
        self,
        feat_dir: str,
        reports_jsonl: str,
        ids: List[str],
        tokenizer: T5Tokenizer,
        max_input_len: int = 48,
        max_target_len: int = 512,
        hint_source: str = "report",  # "report" | "map" | "none"
        organ_map: Optional[Dict[str, str]] = None,
        max_patches: int = 4096,
        subsample_mode: str = "uniform",
    ):
        assert hint_source in {"report", "map", "none"}
        self.feat_dir = pathlib.Path(feat_dir)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.hint_source = hint_source
        self.organ_map = organ_map or {}
        self.max_patches = max_patches
        self.subsample_mode = subsample_mode

        # slide_id -> report
        id2rep: Dict[str, str] = {}
        with open(reports_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sid = obj.get("slide_id")
                rep = obj.get("report")
                if isinstance(sid, str) and isinstance(rep, str) and rep.strip():
                    id2rep[sid] = rep.strip()

        items = []
        for sid in ids:
            # expect file: <sid>.npz under feat_dir (processed_WSI_features)
            # if your extension differs, adapt here
            npz_path = self.feat_dir / f"{sid}.npz"
            if npz_path.exists() and sid in id2rep:
                items.append({"sid": sid, "npz_path": str(npz_path), "report": id2rep[sid]})
        if not items:
            raise ValueError("No matching (patch .npz, report) items found. Check paths and IDs.")
        self.items = items

    @staticmethod
    def make_prompt(organ_hint: Optional[str]) -> str:
        base = ("Generate a formal pathology report for this slide with the following sections: "
                "1) Diagnosis 2) Microscopic Findings 3) Immunohistochemistry "
                "4) Margins and Invasion 5) Staging 6) Additional Notes.")
        if organ_hint:
            return f"{base} Organ/Site: {organ_hint}."
        return base

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        X = load_patch_matrix(it["npz_path"])           # (N, D)
        X = subsample_patches(X, self.max_patches)      # (<=max_patches, D)
        # per-patch L2 norm (stabilizes attention)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        organ_hint = None
        if self.hint_source == "map":
            organ_hint = self.organ_map.get(it["sid"])
        elif self.hint_source == "report":
            organ_hint = _extract_organ_from_text(it["report"])

        prompt = self.make_prompt(organ_hint)

        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_len
        )
        tgt = self.tokenizer(
            it["report"], return_tensors="pt", truncation=True, max_length=self.max_target_len
        )

        return {
            "sid": it["sid"],
            "patches": torch.from_numpy(X.astype("float32")),  # (N, D)
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": tgt.input_ids.squeeze(0),
            "prompt": prompt,
            "organ_hint": organ_hint or "",
        }


def pad_collate_fn(tokenizer: T5Tokenizer):
    def _fn(batch):
        # variable N → pad as a list, pack later inside forward
        patches = [b["patches"] for b in batch]  # list of (N_i, D)
        enc = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "attention_mask": [b["attention_mask"] for b in batch]},
            return_tensors="pt",
        )
        lab = tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        return {
            "patches": patches,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": lab["input_ids"],
            "sid": [b["sid"] for b in batch],
            "prompt": [b["prompt"] for b in batch],
            "organ_hint": [b["organ_hint"] for b in batch],
        }
    return _fn


# -------------------- Metrics helpers --------------------
def clinical_term_recall(pred: str, ref: str) -> float:
    p = pred.lower(); r = ref.lower()
    needed = sum(t in r for t in _TERMS_OF_INTEREST)
    found  = sum((t in p) and (t in r) for t in _TERMS_OF_INTEREST)
    return (found / max(1, needed))


# -------------------- Train --------------------
def main():
    ap = argparse.ArgumentParser()
    # Paths
    ap.add_argument("--feat_dir", required=True, help="Folder with patch-level .npz files")
    ap.add_argument("--reports", required=True)
    ap.add_argument("--splits", required=True)

    # Backbone
    ap.add_argument("--model_name", default="google/flan-t5-base")

    # PMA
    ap.add_argument("--pma_k", type=int, default=12, help="Number of visual tokens after pooling")
    ap.add_argument("--pma_heads", type=int, default=8)
    ap.add_argument("--pma_dropout", type=float, default=0.1)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--subsample_mode", choices=["uniform", "first"], default="uniform")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)

    # Sequence lengths (long enough to learn long reports; adjust per your stats)
    ap.add_argument("--max_input_len", type=int, default=48)
    ap.add_argument("--max_target_len", type=int, default=512)   # learn longish refs
    ap.add_argument("--min_new_tokens", type=int, default=160)   # force substance
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--num_beams", type=int, default=6)
    ap.add_argument("--length_penalty", type=float, default=1.2)

    # Hints
    ap.add_argument("--hint_source", choices=["report", "map", "none"], default="report")
    ap.add_argument("--organ_map_csv", type=str, default="")  # optional CSV with slide_id,organ

    # AMP
    ap.add_argument("--use_amp", action="store_true", default=True)
    ap.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")

    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exp_name", default="flant5base_pma_k12_len512")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer / model
    tok = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Freeze base model; train only LoRA + PMA
    for p in model.parameters():
        p.requires_grad = False
    model.config.dropout_rate = args.dropout
    model.eval().to(device)
    dm = model.config.d_model  # 768 for flan-t5-base

    # LoRA on decoder attention & FFN
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q","k","v","o","wi","wo"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Optional organ CSV
    organ_map = {}
    if args.organ_map_csv:
        df = pd.read_csv(args.organ_map_csv)
        if {"slide_id","organ"}.issubset(df.columns):
            organ_map = dict(zip(df["slide_id"].astype(str), df["organ"].astype(str)))

    # Splits
    sp = json.load(open(args.splits, "r", encoding="utf-8"))

    # Datasets / loaders
    ds_train = SlideReportPMADataset(
        args.feat_dir, args.reports, sp["train"], tok,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len,
        hint_source=args.hint_source, organ_map=organ_map,
        max_patches=args.max_patches, subsample_mode=args.subsample_mode,
    )
    ds_val = SlideReportPMADataset(
        args.feat_dir, args.reports, sp["val"], tok,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len,
        hint_source=args.hint_source, organ_map=organ_map,
        max_patches=args.max_patches, subsample_mode="first",  # deterministic val
    )

    collate = pad_collate_fn(tok)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2, pin_memory=True)

    # PMA pooling module (trainable)
    pma = PMA(in_dim=1024, dm=dm, K=args.pma_k, nheads=args.pma_heads, dropout=args.pma_dropout).to(device)

    # Optimizer: PMA + LoRA params
    trainable_params = list(pma.parameters()) + [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * math.ceil(len(dl_train) / args.grad_accum)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    # Metrics
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    sacrebleu_metric = evaluate.load("sacrebleu")

    # AMP
    use_amp = args.use_amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    def stack_patches(patches_list):
        # Convert list of (N_i, D) arrays to a padded tensor (B, Nmax, D) + mask (B, Nmax)
        Nmax = max(p.shape[0] for p in patches_list)
        B = len(patches_list)
        D = patches_list[0].shape[1]
        X = torch.zeros(B, Nmax, D, dtype=torch.float32)
        M = torch.zeros(B, Nmax, dtype=torch.long)
        for i, p in enumerate(patches_list):
            n = p.shape[0]
            X[i, :n] = p
            M[i, :n] = 1
        return X, M  # (B,Nmax,D), (B,Nmax)

    def train_step(batch) -> torch.Tensor:
        patches, p_mask = stack_patches(batch["patches"])
        patches = patches.to(device)                 # (B,N,D)
        p_mask  = p_mask.to(device)                  # (B,N)
        input_ids = batch["input_ids"].to(device)    # (B,L)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        # PMA pooling → K tokens
        # Masking: MultiheadAttention doesn't accept key_padding_mask with batch_first when using custom inputs_embeds,
        # so we zero out masked patches before projection (already zeros in padded positions).
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                vis_tokens = pma(patches)               # (B,K,dm)
                txt_emb = model.base_model.encoder.embed_tokens(input_ids)  # (B,L,dm)
                enc_inputs = torch.cat([vis_tokens, txt_emb], dim=1)        # (B,K+L,dm)
                vis_mask = torch.ones(patches.size(0), vis_tokens.size(1), dtype=attn_mask.dtype, device=device)
                enc_mask = torch.cat([vis_mask, attn_mask], dim=1)          # (B,K+L)
                out = model(inputs_embeds=enc_inputs, attention_mask=enc_mask, labels=labels)
        else:
            vis_tokens = pma(patches)
            txt_emb = model.base_model.encoder.embed_tokens(input_ids)
            enc_inputs = torch.cat([vis_tokens, txt_emb], dim=1)
            vis_mask = torch.ones(patches.size(0), vis_tokens.size(1), dtype=attn_mask.dtype, device=device)
            enc_mask = torch.cat([vis_mask, attn_mask], dim=1)
            out = model(inputs_embeds=enc_inputs, attention_mask=enc_mask, labels=labels)

        return out.loss

    def validate() -> Tuple[float, float, float, float]:
        pma.eval()
        all_hyps, all_refs, term_recalls = [], [], []

        with torch.no_grad():
            for batch in dl_val:
                patches, p_mask = stack_patches(batch["patches"])
                patches = patches.to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        vis_tokens = pma(patches)
                        txt_emb = model.base_model.encoder.embed_tokens(input_ids)
                        enc_inputs = torch.cat([vis_tokens, txt_emb], dim=1)
                        vis_mask = torch.ones(patches.size(0), vis_tokens.size(1), dtype=attn_mask.dtype, device=device)
                        enc_mask = torch.cat([vis_mask, attn_mask], dim=1)
                        gen = model.generate(
                            inputs_embeds=enc_inputs,
                            attention_mask=enc_mask,
                            min_new_tokens=args.min_new_tokens,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            no_repeat_ngram_size=4,
                        )
                else:
                    vis_tokens = pma(patches)
                    txt_emb = model.base_model.encoder.embed_tokens(input_ids)
                    enc_inputs = torch.cat([vis_tokens, txt_emb], dim=1)
                    vis_mask = torch.ones(patches.size(0), vis_tokens.size(1), dtype=attn_mask.dtype, device=device)
                    enc_mask = torch.cat([vis_mask, attn_mask], dim=1)
                    gen = model.generate(
                        inputs_embeds=enc_inputs,
                        attention_mask=enc_mask,
                        min_new_tokens=args.min_new_tokens,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        no_repeat_ngram_size=4,
                    )

                hyps = tok.batch_decode(gen, skip_special_tokens=True)
                refs = tok.batch_decode(labels, skip_special_tokens=True)

                # Debug peek
                if len(all_hyps) == 0:
                    for j in range(min(2, len(hyps))):
                        print(f"[VAL DEBUG] ref[{j}]: {refs[j][:200]!r}")
                        print(f"[VAL DEBUG] hyp[{j}]: {hyps[j][:200]!r}")

                for h, r in zip(hyps, refs):
                    term_recalls.append(clinical_term_recall(h, r))
                all_hyps.extend(hyps)
                all_refs.extend(refs)

        rougeL = float(evaluate.load("rouge").compute(predictions=all_hyps, references=all_refs)["rougeL"])

        pairs = list(zip(all_hyps, all_refs))
        nonempty = [(h, r) for h, r in pairs if h.strip()]
        if nonempty:
            hne, rne = zip(*nonempty)
            bs = evaluate.load("bertscore").compute(
                predictions=list(hne), references=list(rne), lang="en", rescale_with_baseline=True
            )
            bertF1_ne = float(sum(bs["f1"]) / len(bs["f1"]))
            bertF1 = (bertF1_ne * len(nonempty)) / len(pairs)

            sb = evaluate.load("sacrebleu").compute(
                predictions=list(hne), references=[[r] for r in rne], tokenize="13a"
            )
            bleu_ne = float(sb["score"]) / 100.0
            bleu4 = (bleu_ne * len(nonempty)) / len(pairs)
        else:
            bertF1 = 0.0; bleu4 = 0.0

        termrec = float(sum(term_recalls) / max(1, len(term_recalls)))
        return rougeL, bertF1, bleu4, termrec

    best_rougeL = -1.0
    patience, patience_limit = 0, 7
    save_dir = pathlib.Path("runs") / f"{args.exp_name}-seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        pma.train(True)
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(dl_train, start=1):
            loss = train_step(batch) / args.grad_accum
            loss.backward()
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step(); sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            running += loss.item()

        rougeL, bertF1, bleu4, termrec = validate()
        print(f"Epoch {epoch:02d}: train_loss={running/len(dl_train):.4f}  "
              f"val_ROUGE-L={rougeL:.4f}  val_BERTScoreF1={bertF1:.4f}  "
              f"val_BLEU4={bleu4:.4f}  val_TERMREC={termrec:.3f}",
              flush=True)

        if rougeL > best_rougeL:
            best_rougeL = rougeL; patience = 0
            torch.save(
                {
                    "pma": pma.state_dict(),
                    "best_rougeL": best_rougeL,
                    "peft_config": lora_cfg.to_dict(),
                    "model_name": args.model_name,
                    "pma_k": args.pma_k,
                },
                save_dir / "best.ckpt",
            )
            model.save_pretrained(save_dir / "lora_adapters")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping (no ROUGE-L improvement).", flush=True)
                break

    print(f"Best ROUGE-L: {best_rougeL:.4f}", flush=True)


if __name__ == "__main__":
    main()
