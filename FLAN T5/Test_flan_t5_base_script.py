#!/usr/bin/env python3
import os, sys, json, argparse, glob, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import List, Dict
import evaluate

warnings.filterwarnings("ignore", category=UserWarning)

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_npz_features(p: Path) -> np.ndarray:
    data = np.load(p, allow_pickle=True)
    for k in ("feats", "features", "arr_0", "slide_features"):
        if k in data:
            x = data[k]; break
    else:
        x = data[list(data.keys())[0]]
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x

def jaccard_index(a: str, b: str) -> float:
    sa = set(a.lower().split()); sb = set(b.lower().split())
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

# -----------------------------
# Frozen Attention Pool
# -----------------------------
class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.key = nn.Linear(in_dim, hidden)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [N, D]
        K = self.key(feats)                                   # [N, H]
        q = self.query.expand(1, 1, K.size(-1))               # [1,1,H]
        attn = torch.softmax(torch.matmul(q, K.transpose(0,1)), dim=-1)  # [1,1,N]
        pooled = torch.matmul(attn, feats.unsqueeze(0))       # [1,1,D]
        return pooled.squeeze(0).squeeze(0)                   # [D]

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, d_model: int, prefix_len: int = 8):
        super().__init__()
        self.prefix_len = prefix_len
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.Tanh())

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        h = self.proj(pooled)              # [d_model]
        h = h.unsqueeze(0).repeat(self.prefix_len, 1)
        return h

# ---- Safe checkpoint loader for PyTorch 2.6+ ----
def safe_load_ckpt(ckpt_path: Path):
    from torch.serialization import add_safe_globals
    try:
        try:
            from peft.utils.peft_types import PeftType
            add_safe_globals([PeftType])
        except Exception:
            pass
        import numpy as np
        add_safe_globals([np.dtype])
    except Exception:
        pass
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e1:
        print(f"[WARN] Safe weights-only load failed: {e1}\n"
              f"       Falling back to weights_only=False (trusted checkpoint).", flush=True)
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

def load_attnpool_and_prefix_from_ckpt(ckpt_path: Path, feat_dim: int, d_model: int, prefix_len: int, device: torch.device):
    attnpool = AttentionPool(in_dim=feat_dim).to(device)
    projector = PrefixProjector(in_dim=feat_dim, d_model=d_model, prefix_len=prefix_len).to(device)

    if ckpt_path is None or not ckpt_path.exists():
        log("[WARN] No checkpoint provided or not found. Using randomly initialized attnpool/projector.")
        attnpool.eval(); projector.eval()
        for p in attnpool.parameters(): p.requires_grad = False
        for p in projector.parameters(): p.requires_grad = False
        return attnpool, projector

    ckpt = safe_load_ckpt(ckpt_path)
    state = ckpt.get("state_dict", ckpt)

    ap_sd = {k.split("attnpool.",1)[1]: v for k, v in state.items() if k.startswith("attnpool.")}
    pj_sd = {}
    for prefix in ("prefix_proj.", "projector.", "prefix_mlp.", "proj."):
        pj_sd.update({k.split(prefix,1)[1]: v for k, v in state.items() if k.startswith(prefix)})

    if ap_sd:
        attnpool.load_state_dict(ap_sd, strict=False); log("[INFO] Loaded AttentionPool weights from checkpoint.")
    else:
        log("[WARN] No 'attnpool' weights in checkpoint; using random init.")

    if pj_sd:
        projector.load_state_dict(pj_sd, strict=False); log("[INFO] Loaded Prefix projector weights from checkpoint.")
    else:
        log("[WARN] No projector weights in checkpoint; using random init.")

    attnpool.eval(); projector.eval()
    for p in attnpool.parameters(): p.requires_grad = False
    for p in projector.parameters(): p.requires_grad = False
    return attnpool, projector

@torch.no_grad()
def generate_from_features(model, tokenizer, attnpool, projector, feats_np: np.ndarray, device: torch.device,
                           max_new_tokens: int = 256, num_beams: int = 4, do_sample: bool = False,
                           temperature: float = 1.0):
    feats = torch.from_numpy(feats_np).to(device).float()          # [N, D]
    pooled = attnpool(feats)                                       # [D]
    enc_tokens = projector(pooled).unsqueeze(0)                    # [1, prefix_len, d_model]
    from transformers.modeling_outputs import BaseModelOutput
    enc_out = BaseModelOutput(last_hidden_state=enc_tokens)

    gen_ids = model.generate(
        encoder_outputs=enc_out,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        num_beams=num_beams,
        length_penalty=1.0,
        early_stopping=True
    )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", type=str, required=True, help="Directory with test NPZ features")
    ap.add_argument("--test_jsonl", type=str, required=True, help="Path to test.jsonl")
    ap.add_argument("--adapter_dir", type=str, required=True, help="Folder with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--ckpt", type=str, default=None, help="(Optional) best.ckpt to load attnpool/prefix")
    ap.add_argument("--prefix_len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", help="Enable sampling instead of pure beam search")
    ap.add_argument("--disable_bertscore", action="store_true", help="Skip BERTScore to avoid large model download")
    ap.add_argument("--heartbeat_every", type=int, default=10, help="Print a heartbeat every N samples")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    # Make logs unbuffered even if PYTHONUNBUFFERED not set
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, int(os.environ.get("CPU_THREADS", "1"))))
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] Device          : {device}")
    if device.type == "cuda":
        log(f"[INFO] CUDA name      : {torch.cuda.get_device_name(0)}")
        log(f"[INFO] CUDA capability: {torch.cuda.get_device_capability(0)}")

    feat_dir = Path(args.feat_dir)
    test_jsonl = Path(args.test_jsonl)
    adapter_dir = Path(args.adapter_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = "google/flan-t5-base"
    log("[STEP] Loading tokenizer and base model…")
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    model.to(device)
    d_model = model.config.d_model
    log(f"[INFO] d_model={d_model}")

    # Collect features present
    npz_paths = {Path(p).stem: Path(p) for p in glob.glob(str(feat_dir / "*.npz"))}
    log(f"[INFO] Found {len(npz_paths)} feature files in {feat_dir}")

    # Load test.jsonl -> id->ref
    id2ref: Dict[str, str] = {}
    with open(test_jsonl, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            sid = row.get("id") or row.get("slide_id") or row.get("slide") or row.get("wsi_id")
            ref = row.get("report") or row.get("text") or row.get("gt") or row.get("reference")
            if sid is None or ref is None: continue
            id2ref[str(sid)] = str(ref)

    ids_available = sorted(set(npz_paths.keys()) & set(id2ref.keys()))
    if len(ids_available) == 0:
        log("[WARN] No overlap between features in feat_dir and entries in test.jsonl.")
        (out_dir / "predictions.jsonl").write_text("")
        (out_dir / "metrics_summary.json").write_text(json.dumps({"count": 0}))
        return

    # Determine feature dimension from first file
    first_feats = load_npz_features(npz_paths[ids_available[0]])
    feat_dim = int(first_feats.shape[-1])
    log(f"[INFO] Feature dim      : {feat_dim}")
    assert first_feats.ndim == 2, f"Features must be [N,D], got {first_feats.shape}"

    # Load pooling + projector
    log("[STEP] Loading AttentionPool + Prefix projector…")
    attnpool, projector = load_attnpool_and_prefix_from_ckpt(ckpt_path, feat_dim, d_model, args.prefix_len, device)

    # Metrics
    log("[STEP] Initializing metrics…")
    rouge = evaluate.load("rouge")
    bertscore = None
    if not args.disable_bertscore:
        # WARNING: downloads a large model if not cached; can stall on clusters without egress.
        try:
            bertscore = evaluate.load("bertscore")
        except Exception as e:
            log(f"[WARN] Could not load BERTScore evaluator ({e}). Continuing without it.")
            bertscore = None


    # BLEU smoothing
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method3
    from nltk.translate.meteor_score import single_meteor_score

    preds_path = out_dir / "predictions.jsonl"

    cases = []            # to hold per-case metrics
    all_refs = []
    all_hyps = []

    log(f"[STEP] Testing on {len(ids_available)} slides…")
    start_time = time.time()

    for i, sid in enumerate(ids_available, 1):
        feats_np = load_npz_features(npz_paths[sid])
        if feats_np.shape[-1] != feat_dim:
            raise ValueError(f"Dim mismatch for {sid}: got {feats_np.shape[-1]} vs {feat_dim}")

        hyp = generate_from_features(
            model, tokenizer, attnpool, projector, feats_np, device,
            max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
            do_sample=args.sample, temperature=args.temperature
        )
        ref = id2ref[sid]

        # tokens
        ref_tok = ref.split()
        hyp_tok = hyp.split()

        # BLEU 1..4
        bleu1 = sentence_bleu([ref_tok], hyp_tok, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
        bleu2 = sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
        bleu3 = sentence_bleu([ref_tok], hyp_tok, weights=(1/3, 1/3, 1/3, 0.0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        # METEOR
        try:
            meteor = float(single_meteor_score(ref, hyp))
        except Exception as e:
            log(f"[WARN] METEOR failed for id={sid} ({e}). Using 0.0.")
            meteor = 0.0

        # Jaccard
        jac = jaccard_index(ref, hyp)

        cases.append({
            "id": sid,
            "reference": ref,
            "prediction": hyp,
            "BLEU1": round(float(bleu1), 6),
            "BLEU2": round(float(bleu2), 6),
            "BLEU3": round(float(bleu3), 6),
            "BLEU4": round(float(bleu4), 6),
            "METEOR": round(float(meteor), 6),
            "Jaccard": round(float(jac), 6),
        })

        all_refs.append(ref); all_hyps.append(hyp)

        if i % max(1, args.heartbeat_every) == 0:
            elapsed = time.time() - start_time
            log(f"[HB] {i}/{len(ids_available)} done | elapsed={elapsed/60:.1f}m")

    # Aggregate metrics (ROUGE & BERTScore)
    log("[STEP] Computing aggregate & per-case metrics…")
    if rouge is not None:
        rouge_per = rouge.compute(predictions=all_hyps, references=all_refs,
                                  use_stemmer=True, rouge_types=["rougeL"], use_aggregator=False)
        rougeL_list = rouge_per.get("rougeL", [0.0]*len(all_refs))
        rouge_agg = rouge.compute(predictions=all_hyps, references=all_refs, use_stemmer=True)
        rougeL_avg = float(rouge_agg.get("rougeL", 0.0))
    else:
        rougeL_list = [0.0]*len(all_refs); rougeL_avg = 0.0

    if bertscore is not None:
        try:
            bert_res = bertscore.compute(
                predictions=all_hyps, references=all_refs,
                model_type=os.environ.get("BERTSCORE_MODEL", "microsoft/deberta-large-mnli")
            )
            bert_f1_list = [float(x) for x in bert_res.get("f1", [0.0]*len(all_refs))]
            bert_f1_avg = float(np.mean(bert_f1_list)) if bert_f1_list else 0.0
        except Exception as e:
            log(f"[WARN] BERTScore failed ({e}). Setting to zeros.")
            bert_f1_list = [0.0]*len(all_refs); bert_f1_avg = 0.0
    else:
        bert_f1_list = [0.0]*len(all_refs); bert_f1_avg = 0.0

    # merge per-case ROUGE_L and BERTScore_F1, then save JSONL
    for idx, c in enumerate(cases):
        c["ROUGE_L"] = round(float(rougeL_list[idx]), 6)
        c["BERTScore_F1"] = round(float(bert_f1_list[idx]), 6)

    with open(preds_path, "w") as preds_f:
        for c in cases:
            preds_f.write(json.dumps(c) + "\\n")
    log(f"[INFO] Predictions (with per-case metrics) saved to: {preds_path}")

    # Averages
    import numpy as _np
    metrics_summary = {
        "count": len(cases),
        "BLEU1_avg": float(_np.mean([c["BLEU1"] for c in cases])) if cases else 0.0,
        "BLEU2_avg": float(_np.mean([c["BLEU2"] for c in cases])) if cases else 0.0,
        "BLEU3_avg": float(_np.mean([c["BLEU3"] for c in cases])) if cases else 0.0,
        "BLEU4_avg": float(_np.mean([c["BLEU4"] for c in cases])) if cases else 0.0,
        "METEOR_avg": float(_np.mean([c["METEOR"] for c in cases])) if cases else 0.0,
        "ROUGE_L_avg": rougeL_avg,
        "Jaccard_avg": float(_np.mean([c["Jaccard"] for c in cases])) if cases else 0.0,
        "BERTScore_F1_avg": bert_f1_avg,
    }

    with open(Path(args.out_dir) / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    log("\\n=== Metrics (averaged) ===")
    log(f"Samples            : {metrics_summary['count']}")
    log(f"BLEU-1 (avg)       : {metrics_summary['BLEU1_avg']:.4f}")
    log(f"BLEU-2 (avg)       : {metrics_summary['BLEU2_avg']:.4f}")
    log(f"BLEU-3 (avg)       : {metrics_summary['BLEU3_avg']:.4f}")
    log(f"BLEU-4 (avg)       : {metrics_summary['BLEU4_avg']:.4f}")
    log(f"METEOR (avg)       : {metrics_summary['METEOR_avg']:.4f}")
    log(f"ROUGE-L (avg)      : {metrics_summary['ROUGE_L_avg']:.4f}")
    log(f"BERTScore F1 (avg) : {metrics_summary['BERTScore_F1_avg']:.4f}")
    log(f"Jaccard (avg)      : {metrics_summary['Jaccard_avg']:.4f}")
    log(f"Summary saved to   : {Path(args.out_dir)/'metrics_summary.json'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # make sure we fail loudly in Slurm logs
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        raise


def _finalize_and_score_predictions(ids_available, id2ref, npz_paths, feat_dim,
                                    model, tokenizer, attnpool, projector, device, args, out_dir, log):
    import json, os, time
    import numpy as np
    from pathlib import Path
    preds_path = out_dir / "predictions.jsonl"

    cases = []
    all_refs = []
    all_hyps = []

    log(f"[STEP] Testing on {len(ids_available)} slides…")
    start_time = time.time()

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, sentence_bleu
    smooth = SmoothingFunction().method3

    for i, sid in enumerate(ids_available, 1):
        feats_np = load_npz_features(npz_paths[sid])
        if feats_np.shape[-1] != feat_dim:
            raise ValueError(f"Dim mismatch for {sid}: got {feats_np.shape[-1]} vs {feat_dim}")

        hyp = generate_from_features(
            model, tokenizer, attnpool, projector, feats_np, device,
            max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
            do_sample=args.sample, temperature=args.temperature
        )
        ref = id2ref[sid]

        ref_tok = ref.split()
        hyp_tok = hyp.split()

        bleu1 = sentence_bleu([ref_tok], hyp_tok, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
        bleu2 = sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
        bleu3 = sentence_bleu([ref_tok], hyp_tok, weights=(1/3, 1/3, 1/3, 0.0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        try:
            meteor = float(single_meteor_score(ref, hyp))
        except Exception:
            meteor = 0.0

        jac = jaccard_index(ref, hyp)

        cases.append({
            "id": sid,
            "reference": ref,
            "prediction": hyp,
            "BLEU1": round(float(bleu1), 6),
            "BLEU2": round(float(bleu2), 6),
            "BLEU3": round(float(bleu3), 6),
            "BLEU4": round(float(bleu4), 6),
            "METEOR": round(float(meteor), 6),
            "Jaccard": round(float(jac), 6),
        })

        all_refs.append(ref)
        all_hyps.append(hyp)

        if i % max(1, args.heartbeat_every) == 0:
            elapsed = time.time() - start_time
            log(f"[HB] {i}/{len(ids_available)} done | elapsed={elapsed/60:.1f}m")

    # ROUGE-L
    if rouge is not None:
        rouge_per = rouge.compute(predictions=all_hyps, references=all_refs,
                                  use_stemmer=True, rouge_types=["rougeL"], use_aggregator=False)
        rougeL_list = rouge_per.get("rougeL", [0.0]*len(all_refs))
        rouge_agg = rouge.compute(predictions=all_hyps, references=all_refs, use_stemmer=True)
        rougeL_avg = float(rouge_agg.get("rougeL", 0.0))
    else:
        rougeL_list = [0.0]*len(all_refs)
        rougeL_avg = 0.0

    # BERTScore
    if bertscore is not None:
        try:
            bert_res = bertscore.compute(
                predictions=all_hyps, references=all_refs,
                model_type=os.environ.get("BERTSCORE_MODEL", "microsoft/deberta-large-mnli")
            )
            bert_f1_list = [float(x) for x in bert_res.get("f1", [0.0]*len(all_refs))]
            bert_f1_avg = float(np.mean(bert_f1_list)) if bert_f1_list else 0.0
        except Exception:
            bert_f1_list = [0.0]*len(all_refs)
            bert_f1_avg = 0.0
    else:
        bert_f1_list = [0.0]*len(all_refs)
        bert_f1_avg = 0.0

    for idx, c in enumerate(cases):
        c["ROUGE_L"] = round(float(rougeL_list[idx]), 6)
        c["BERTScore_F1"] = round(float(bert_f1_list[idx]), 6)

    with open(preds_path, "w") as preds_f:
        for c in cases:
            preds_f.write(json.dumps(c) + "\n")

    import numpy as _np
    metrics_summary = {
        "count": len(cases),
        "BLEU1_avg": float(_np.mean([c["BLEU1"] for c in cases])) if cases else 0.0,
        "BLEU2_avg": float(_np.mean([c["BLEU2"] for c in cases])) if cases else 0.0,
        "BLEU3_avg": float(_np.mean([c["BLEU3"] for c in cases])) if cases else 0.0,
        "BLEU4_avg": float(_np.mean([c["BLEU4"] for c in cases])) if cases else 0.0,
        "METEOR_avg": float(_np.mean([c["METEOR"] for c in cases])) if cases else 0.0,
        "ROUGE_L_avg": float(rougeL_avg),
        "Jaccard_avg": float(_np.mean([c["Jaccard"] for c in cases])) if cases else 0.0,
        "BERTScore_F1_avg": float(bert_f1_avg),
    }

    with open(Path(args.out_dir) / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    log("\\n=== Metrics (averaged) ===")
    log(f"Samples            : {metrics_summary['count']}")
    log(f"BLEU-1 (avg)       : {metrics_summary['BLEU1_avg']:.4f}")
    log(f"BLEU-2 (avg)       : {metrics_summary['BLEU2_avg']:.4f}")
    log(f"BLEU-3 (avg)       : {metrics_summary['BLEU3_avg']:.4f}")
    log(f"BLEU-4 (avg)       : {metrics_summary['BLEU4_avg']:.4f}")
    log(f"METEOR (avg)       : {metrics_summary['METEOR_avg']:.4f}")
    log(f"ROUGE-L (avg)      : {metrics_summary['ROUGE_L_avg']:.4f}")
    log(f"BERTScore F1 (avg) : {metrics_summary['BERTScore_F1_avg']:.4f}")
    log(f"Jaccard (avg)      : {metrics_summary['Jaccard_avg']:.4f}")
    log(f"Summary saved to   : {Path(args.out_dir)/'metrics_summary.json'}")
