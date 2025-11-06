#!/usr/bin/env python
import os, json, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score


# ------------------------------
# Minimal Attention Pool (frozen)
# ------------------------------
class AttentionPool(nn.Module):
    """
    Frozen single-head attention pooling over tile features.
    Expects features shaped (num_tiles, dim). Returns (dim,).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.q, mean=0.0, std=1.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [N, D]
        if feats.ndim != 2:
            feats = feats.view(feats.shape[0], -1)
        scores = torch.mv(feats, self.q) / math.sqrt(feats.size(1))
        w = torch.softmax(scores, dim=0)  # [N]
        pooled = torch.matmul(w, feats)   # [D]
        return pooled


# ------------------------------
# Dataset & Collate
# ------------------------------
class WSITestDataset(Dataset):
    """
    Loads only the slide ids present in test.jsonl that also have .npz features.
    """
    def __init__(self, feat_dir: Path, test_jsonl: Path):
        self.feat_dir = Path(feat_dir)
        self.items = []  # list of dicts: {id, ref, npz}

        with open(test_jsonl, "r") as f:
            rows = [json.loads(l) for l in f]

        # Each row must have "npz_id" and "report"
        for r in rows:
            sid = r.get("npz_id") or r.get("slide_id") or r.get("id")
            rep = r.get("report", "")
            if sid is None:
                continue
            npz_path = self.feat_dir / f"{sid}.npz"
            if npz_path.exists():
                self.items.append({"id": sid, "ref": rep, "npz": npz_path})

        if len(self.items) == 0:
            print("[WARN] No overlapping slide_ids between features and test.jsonl.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        arr = np.load(rec["npz"])
        # support either 'features' or the first array key
        if "features" in arr:
            feats = arr["features"]
        else:
            # take first array in file
            k = list(arr.keys())[0]
            feats = arr[k]
        # ensure 2D: [tiles, dim]
        feats = feats.reshape(-1, feats.shape[-1]) if feats.ndim > 2 else feats
        return rec["id"], feats.astype(np.float32), rec["ref"]


class CollateCPU:
    """
    - Pools raw tile features -> single slide vector
    - Tokenizes text target (references) for metrics (kept only for alignment)
    """
    def __init__(self, tokenizer: AutoTokenizer, pool: AttentionPool, device: str = "cpu"):
        self.tok = tokenizer
        self.pool = pool
        self.device = torch.device(device)

    def __call__(self, batch):
        ids, pooled, refs = [], [], []
        with torch.no_grad():
            for sid, feats, ref in batch:
                t = torch.from_numpy(feats)  # [N,D] on CPU
                vec = self.pool(t).float()   # [D] on CPU
                pooled.append(vec)
                ids.append(sid)
                refs.append(ref)

        pooled = torch.stack(pooled, dim=0)  # [B, D]
        return {
            "slide_ids": ids,
            "pooled": pooled,  # CPU tensor
            "references": refs
        }


# ------------------------------
# Metrics
# ------------------------------
def jaccard_index(preds: List[str], refs: List[str]) -> float:
    scores = []
    for p, r in zip(preds, refs):
        ps, rs = set(p.split()), set(r.split())
        u = len(ps.union(rs))
        i = len(ps.intersection(rs))
        scores.append(0.0 if u == 0 else i / u)
    return float(np.mean(scores)) if scores else 0.0


def compute_all_metrics(refs: List[str], preds: List[str]) -> Dict[str, float]:
    out = {}
    # ROUGE-L
    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    out["rougeL_f1"] = float(r["rougeL"].mid.fmeasure) if hasattr(r["rougeL"], "mid") else float(r["rougeL"])

    # BLEU-4 (sentence-level average with smoothing)
    chencherry = SmoothingFunction()
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=chencherry.method3)
        for ref, pred in zip(refs, preds)
    ]
    out["bleu4"] = float(np.mean(bleu_scores)) if bleu_scores else 0.0

    # BERTScore (F1)
    _, _, F1 = bert_score.score(preds, refs, lang="en", rescale_with_baseline=True)
    out["bertscore_f1"] = float(F1.mean().cpu().numpy())

    # Jaccard
    out["jaccard"] = jaccard_index(preds, refs)
    return out


# ------------------------------
# Model build & ckpt load (PEFT)
# ------------------------------
def build_frozen_biobart_with_lora(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    device: str
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.eval()
    model.to(device)
    return model, tok


def load_lightning_peft_ckpt(peft_model: nn.Module, ckpt_path: Path):
    """
    Robustly load a PL .ckpt that contains LoRA + base weights under keys like:
      - 'state_dict' OR 'model'
      - prefixed with 'base_model.' ... including LoRA tensors.
    We load with strict=False to ignore any trainer/scalar keys.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and ("state_dict" in sd):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and ("model" in sd):
        sd = sd["model"]

    # Try PEFT helper first (best for adapter weights). If it fails, fall back.
    try:
        set_peft_model_state_dict(peft_model, sd, adapter_name="default")
    except Exception:
        peft_model.load_state_dict(sd, strict=False)


# ------------------------------
# Inference (generate from pooled vector)
# ------------------------------
class LinearProjector(nn.Module):
    """
    Projects pooled vector -> pseudo "prompt" tokens for encoder.
    Simple approach: map D -> (T, hidden) and feed as embeddings.
    """
    def __init__(self, in_dim: int, hidden_dim: int, T: int = 8):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(in_dim, hidden_dim * T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] -> [B, T, H]
        b = x.size(0)
        y = self.proj(x)            # [B, T*H]
        y = y.view(b, self.T, -1)   # [B, T, H]
        return y


@torch.no_grad()
def generate_reports(
    model, tok, projector, batch, device, max_new_tokens=256
) -> List[str]:
    pooled = batch["pooled"].to(device)  # [B, D]
    # Project pooled features to encoder hidden size (BioBART base: d_model=768)
    encoder_embeds = projector(pooled)   # [B, T, 768]

    # Use model.generate with encoder_outputs
    encoder_outputs = model.get_encoder()(inputs_embeds=encoder_embeds)
    gen_ids = model.generate(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=torch.full((encoder_embeds.size(0), 1),
                                     tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id,
                                     dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
    return texts


# ------------------------------
# Main
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, required=True,
                    help="Folder containing .npz tile features named {slide_id}.npz")
    ap.add_argument("--test_jsonl", type=str, required=True,
                    help="JSONL with fields: npz_id (or id/slide_id) and report")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to Lightning ckpt (best.ckpt)")
    ap.add_argument("--model_name", type=str, default="GanjinZero/biobart-v2-base")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str,
                    default="q_proj,k_proj,v_proj,out_proj,fc1,fc2")
    ap.add_argument("--proj_tokens", type=int, default=8,
                    help="Number of pseudo-tokens fed to encoder")
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Build frozen BioBART (+ LoRA wrapper) and tokenizer
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model, tok = build_frozen_biobart_with_lora(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout, target_modules, device
    )

    # Load Lightning ckpt into PEFT model (includes LoRA weights)
    load_lightning_peft_ckpt(model, Path(args.checkpoint))

    # Set up frozen attention pool + projector
    # Infer feature dim from one file
    ds_tmp = WSITestDataset(Path(args.features_dir), Path(args.test_jsonl))
    if len(ds_tmp) == 0:
        print("[ERROR] No usable test items. Check npz ids vs test.jsonl.")
        return
    sample_id, sample_feats, _ = ds_tmp[0]
    feat_dim = int(sample_feats.shape[1])

    attnpool = AttentionPool(feat_dim)
    for p in attnpool.parameters():
        p.requires_grad = False
    attnpool.eval()

    # BioBART base hidden size = config.d_model
    hidden_dim = int(getattr(model.base_model, "model").config.d_model)
    projector = LinearProjector(in_dim=feat_dim, hidden_dim=hidden_dim, T=args.proj_tokens).to(device)
    for p in projector.parameters():
        p.requires_grad = False
    projector.eval()

    # Data
    ds = ds_tmp
    collate = CollateCPU(tok, attnpool, device="cpu")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, pin_memory=False)

    all_refs, all_hyps, all_ids = [], [], []
    for batch in dl:
        hyps = generate_reports(model, tok, projector, batch, device, args.max_new_tokens)
        refs = batch["references"]
        ids  = batch["slide_ids"]
        all_hyps.extend(hyps)
        all_refs.extend(refs)
        all_ids.extend(ids)

    # Metrics
    metrics = compute_all_metrics(all_refs, all_hyps)

    # Save predictions.jsonl
    pred_path = Path(args.out_dir) / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for sid, ref, hyp in zip(all_ids, all_refs, all_hyps):
            f.write(json.dumps({"slide_id": sid, "reference": ref, "prediction": hyp}) + "\n")

    # Save metrics (json + nice table)
    with open(Path(args.out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Pretty print table & averages
    print("\n=== Metrics (test set) ===")
    print(f"ROUGE-L (F1): {metrics['rougeL_f1']:.4f}")
    print(f"BLEU-4     : {metrics['bleu4']:.4f}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"Jaccard    : {metrics['jaccard']:.4f}")
    print(f"\nSaved: {pred_path} and metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
