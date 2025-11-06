#!/usr/bin/env python3
import os, json, math, argparse, pathlib, gc, csv, random, re
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          get_linear_schedule_with_warmup, GenerationConfig)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate

# ---------------- Utils ----------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def is_hf_cache_snapshot_path(p:str)->bool:
    return isinstance(p,str) and bool(re.search(r"/models--[^/]+--[^/]+/snapshots/[0-9a-f]{16,}", p))

def is_valid_model_dir(p:str)->bool:
    d=pathlib.Path(p)
    if not d.is_dir(): return False
    return any((d/f).exists() for f in ["config.json","tokenizer_config.json","pytorch_model.bin","model.safetensors"])

def try_load_tokenizer_and_model(repo_id:str, token:Optional[str], local_only:bool):
    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True,
                                        token=token, local_files_only=local_only)
    # keep model base dtype = float32 for stability with LoRA adapters
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        token=token,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    return tok, model

def build_model_exact(model_name:str):
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if is_hf_cache_snapshot_path(model_name):
        raise ValueError(
            f"You passed a Hugging Face cache snapshot path:\n  {model_name}\n"
            "Please pass an official repo id (e.g. 'facebook/bart-base') or a real local directory "
            "containing model files (config.json, tokenizer_config.json, weights)."
        )
    if is_valid_model_dir(model_name):
        tok, model = try_load_tokenizer_and_model(model_name, token, local_only=True)
        return model, tok
    tok, model = try_load_tokenizer_and_model(model_name, token, local_only=False)
    return model, tok

def load_reports(jsonl_path:str)->Dict[str,str]:
    d={}
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            sid=obj.get("slide_id") or obj.get("id")
            rpt=obj.get("report") or obj.get("text") or obj.get("target")
            if sid and rpt: d[sid]=rpt.strip()
    return d

def discover_feature(path:pathlib.Path)->Optional[Tuple[str,pathlib.Path]]:
    return (path.stem, path) if path.suffix.lower() in (".npz",".npy") else None

def load_patches(p:pathlib.Path)->np.ndarray:
    if p.suffix.lower()==".npy":
        arr=np.load(p,allow_pickle=False)
        if arr.ndim==1: return arr[None,:].astype(np.float32)
        if arr.ndim==2: return arr.astype(np.float32)
        return arr.reshape(arr.shape[0],-1).astype(np.float32)
    z=np.load(p,allow_pickle=False)
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
    raise ValueError(f"No 2D patches in {p.name}")

def parse_splits(path:str)->Dict[str,List[str]]:
    with open(path,"r") as f: return json.load(f)

def load_organ_map(csv_path:str)->Dict[str,Dict[str,str]]:
    m={}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid=row.get("slide_id") or row.get("id")
            if not sid: continue
            m[sid]={"organ":(row.get("organ") or "").strip(),
                    "site_detail":(row.get("site_detail") or "").strip()}
    return m

def load_lexicon(csv_path: Optional[str])->Dict[str,List[str]]:
    if not csv_path: return {}
    store={}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            org=(row.get("organ") or "").strip()
            term=(row.get("term") or "").strip()
            try: w=float(row.get("weight","1"))
            except: w=1.0
            if org and term: store.setdefault(org,[]).append((w,term))
    out={}
    for org,pairs in store.items():
        pairs.sort(key=lambda x:-x[0])
        seen=set(); terms=[]
        for w,t in pairs:
            tl=t.lower()
            if tl in seen: continue
            seen.add(tl); terms.append(t)
        out[org]=terms[:20]
    return out

CHECKLIST_BY_ORGAN={
    "breast":["tumor size (cm)","histologic type & grade","ER/PR/HER2","margins (mm)","LVI/PNI","nodes (counts, ENE)","stage"],
    "colon":["site","size","grade","depth (T)","LVI/PNI","margins","nodes","MMR/MSI","stage"],
    "rectum":["size","grade","CRM & distal margins","T category","LVI/PNI","nodes","MMR/MSI","stage"],
    "stomach":["size","Lauren type/grade","serosal/adjacent organ invasion","margins","nodes","HER2","stage"],
    "oral cavity":["subsite","size (cm)","DOI (mm)","grade","margins","PNI/LVI","nodes (levels, ENE)","pT/pN"],
    "lung":["lobe","histology & grade","size","VPI/LVI/STAS","margins","nodes","PD-L1","pT/pN"],
    "lymph node":["subtype","IHC panel","Ki-67 (%)","genetics","Lugano stage"],
    "brain":["site/laterality","integrated diagnosis","grade","MIB-1 (%)","necrosis/MVP","extent of resection"],
    "_default":["site","tumor size","histology & grade","margins","LVI/PNI","nodes","stage"],
}
def build_prompt(organ:str, site_detail:str, terms:List[str])->str:
    org=(organ or "").strip(); site=(site_detail or "").strip()
    ck=CHECKLIST_BY_ORGAN.get(org.lower(), CHECKLIST_BY_ORGAN["_default"])
    tstr="; ".join(terms[:12]) if terms else ""
    head=f"[TASK] Generate a complete pathology report for organ={org or 'unknown'}"
    if site: head+=f" (site={site})"
    parts=[head, "[CHECKLIST] Include: "+", ".join(ck)+".",
           f"[TERMS] Prefer: {tstr}." if tstr else "",
           "[STYLE] Cohesive paragraphs, formal, no patient identifiers."]
    return " ".join([p for p in parts if p]).strip()

# ---------------- PMA Pooler ----------------
class PatchEncoder(nn.Module):
    def __init__(self,in_dim:int,d_model:int):
        super().__init__()
        self.proj=nn.Linear(in_dim,d_model); self.act=nn.GELU(); self.norm=nn.LayerNorm(d_model)
    def forward(self,x): return self.norm(self.act(self.proj(x)))

class TinyTransformer(nn.Module):
    def __init__(self,d_model:int,n_heads:int=8,n_layers:int=2,dropout:float=0.1):
        super().__init__()
        layer=nn.TransformerEncoderLayer(d_model,n_heads,d_model*4,dropout,
                                         activation="gelu",batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,num_layers=n_layers)
    def forward(self,x,key_padding_mask=None):
        return self.enc(x, src_key_padding_mask=key_padding_mask)

class PMA(nn.Module):
    def __init__(self,d_model:int,k_tokens:int=24,n_heads:int=8,dropout:float=0.1):
        super().__init__()
        self.q=nn.Parameter(torch.randn(k_tokens,d_model))
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.ln=nn.LayerNorm(d_model)
    def forward(self,x,key_padding_mask=None):
        B=x.size(0); Q=self.q.unsqueeze(0).expand(B,-1,-1)
        y,_=self.attn(Q,x,x,key_padding_mask=key_padding_mask,need_weights=False)
        return self.ln(y)

class AttnPooler(nn.Module):
    def __init__(self,in_dim:int,d_model:int,k_tokens:int=24,n_heads:int=8,n_layers:int=2,dropout:float=0.1):
        super().__init__()
        self.enc_proj=PatchEncoder(in_dim,d_model)
        self.backbone=TinyTransformer(d_model,n_heads,n_layers,dropout)
        self.pma=PMA(d_model,k_tokens,n_heads,dropout)
        self.post=nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.LayerNorm(d_model))
    def forward(self,patches:torch.Tensor,pad_mask:Optional[torch.Tensor]=None):
        h=self.enc_proj(patches); h=self.backbone(h,key_padding_mask=pad_mask)
        k=self.pma(h,key_padding_mask=pad_mask); return self.post(k)

# ---------------- Data ----------------
class SlidesPatchDataset(Dataset):
    def __init__(self, ids:List[str], id2path:Dict[str,pathlib.Path],
                 id2report:Dict[str,str], organ_map:Dict[str,Dict[str,str]],
                 organ_terms:Dict[str,List[str]], max_patches:int=2048, mode:str="train"):
        self.ids=[i for i in ids if i in id2path and i in id2report]
        self.id2path=id2path; self.id2report=id2report
        self.organ_map=organ_map; self.organ_terms=organ_terms
        self.max_patches=max_patches; self.mode=mode
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sid=self.ids[idx]
        patches=load_patches(self.id2path[sid])
        N=patches.shape[0]
        if self.mode=="train" and N>self.max_patches:
            idxs=np.random.choice(N, self.max_patches, replace=False); patches=patches[idxs]
        elif self.mode!="train" and N>self.max_patches:
            idxs=np.linspace(0, N-1, num=self.max_patches, dtype=int); patches=patches[idxs]
        org_info=self.organ_map.get(sid,{})
        organ=org_info.get("organ",""); site=org_info.get("site_detail","")
        terms=self.organ_terms.get(organ,[])
        prompt=build_prompt(organ, site, terms)
        return {"slide_id":sid, "patches":torch.from_numpy(patches),
                "text_in":prompt, "text_out":self.id2report[sid]}

class CPUCollator:
    def __init__(self, tokenizer, max_source_len:int, label_max_len:int):
        self.tok=tokenizer
        self.max_source_len=max_source_len
        self.label_max_len=label_max_len
    def __call__(self, batch):
        plist=[b["patches"] for b in batch]
        maxN=max(p.shape[0] for p in plist); Din=plist[0].shape[1]; B=len(plist)
        P=torch.zeros(B,maxN,Din,dtype=torch.float32)
        mask=torch.ones(B,maxN,dtype=torch.bool)
        for i,p in enumerate(plist):
            n=p.shape[0]; P[i,:n]=p; mask[i,:n]=False
        texts=[b["text_in"] for b in batch]
        tokk=self.tok(texts, padding=True, truncation=True,
                      max_length=self.max_source_len, return_tensors="pt")
        yk=self.tok(text_target=[b["text_out"] for b in batch],
                    padding=True, truncation=True, max_length=self.label_max_len,
                    return_tensors="pt")
        labels=yk["input_ids"]; labels[labels==self.tok.pad_token_id]=-100
        return {"patches":P, "pad_mask":mask, "text_in_ids": tokk["input_ids"],
                "text_in_attn": tokk["attention_mask"], "labels": labels}

# ---------------- Metrics ----------------
def compute_metrics(refs, hyps):
    rouge=evaluate.load("rouge"); r=rouge.compute(predictions=hyps, references=refs, use_stemmer=True)
    rougeL=r.get("rougeL",0.0)
    bleu=evaluate.load("sacrebleu"); b=bleu.compute(predictions=hyps, references=[[t] for t in refs])
    bleu4=float(b["score"])/100.0
    try:
        bert=evaluate.load("bertscore")
        bb=bert.compute(predictions=hyps, references=refs, model_type="roberta-large")
        bert_f1=float(sum(bb["f1"])/len(bb["f1"]))
    except Exception:
        bert_f1=0.0
    return {"rougeL":rougeL, "bleu4":bleu4, "bertscore_f1":bert_f1}

@torch.no_grad()
def model_decode(tok, labels):
    y = labels.clone(); y[y==-100]=tok.pad_token_id
    return tok.batch_decode(y, skip_special_tokens=True)

# ---------------- Model ----------------
def build_model(model_name:str, lora_r:int, lora_alpha:int, lora_dropout:float, grad_ckpt:bool):
    model, tok = build_model_exact(model_name)
    if model.get_input_embeddings().weight.shape[0] != len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    peft_cfg=LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","out_proj"]
    )
    model=get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tok

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--reports_jsonl", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--organ_csv", required=True)
    ap.add_argument("--lexicon_csv", default="")
    ap.add_argument("--text_model_name", default="facebook/bart-base")

    # Pooler
    ap.add_argument("--k_prefix", type=int, default=24)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--max_patches", type=int, default=2048)

    # Lengths
    ap.add_argument("--max_source_len", type=int, default=1024)
    ap.add_argument("--label_max_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=768)
    ap.add_argument("--min_decode_len", type=int, default=300)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--length_penalty", type=float, default=1.15)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)

    # Train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_bsz", type=int, default=1)
    ap.add_argument("--eval_bsz", type=int, default=1)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--warmup_steps", type=int, default=800)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.10)

    # LoRA / CKPT
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.15)
    ap.add_argument("--grad_checkpoint", action="store_true")

    ap.add_argument("--out_dir", default="runs/bart_attnpool")
    ap.add_argument("--seed", type=int, default=13)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    feat_dir=pathlib.Path(args.feat_dir)
    id2path={}
    for p in feat_dir.iterdir():
        info=discover_feature(p)
        if info:
            sid,path=info; id2path[sid]=path
    if not id2path:
        raise RuntimeError(f"No .npz/.npy found in {feat_dir}")

    id2report=load_reports(args.reports_jsonl)
    splits=parse_splits(args.splits)
    organ_map=load_organ_map(args.organ_csv)
    organ_terms=load_lexicon(args.lexicon_csv) if args.lexicon_csv else {}

    model,tok=build_model(args.text_model_name, args.lora_r, args.lora_alpha,
                          args.lora_dropout, args.grad_checkpoint)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_dtype = model.get_input_embeddings().weight.dtype  # authoritative dtype (Float32 here)

    d_model=model.config.d_model
    in_dim=load_patches(next(iter(id2path.values()))).shape[1]
    pooler=AttnPooler(in_dim, d_model, k_tokens=args.k_prefix,
                      n_heads=args.n_heads, n_layers=args.n_layers,
                      dropout=args.dropout).to(device)

    ds_tr=SlidesPatchDataset(splits["train"], id2path, id2report, organ_map, organ_terms,
                             max_patches=args.max_patches, mode="train")
    ds_va=SlidesPatchDataset(splits.get("val",[]), id2path, id2report, organ_map, organ_terms,
                             max_patches=args.max_patches, mode="val")
    collate=CPUCollator(tok,args.max_source_len,args.label_max_len)
    dl_tr=DataLoader(ds_tr,batch_size=args.train_bsz,shuffle=True,num_workers=4,
                     collate_fn=collate,persistent_workers=False)
    dl_va=DataLoader(ds_va,batch_size=args.eval_bsz,shuffle=False,num_workers=2,
                     collate_fn=collate,persistent_workers=False)

    params=list(pooler.parameters())+[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps=args.epochs*max(1, math.ceil(len(dl_tr)/max(1,args.grad_accum)))
    sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    label_smooth=args.label_smoothing
    step_accum=0; best_rouge=-1.0; patience=4; no_improve=0

    def make_inputs(batch):
        P = batch["patches"].to(device, non_blocking=True).to(model_dtype)
        mask = batch["pad_mask"].to(device, non_blocking=True)
        text_ids = batch["text_in_ids"].to(device, non_blocking=True)
        text_attn = batch["text_in_attn"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        prefix = pooler(P, pad_mask=mask)  # [B,K,d] float32
        prefix = prefix.to(model_dtype)

        K = prefix.size(1)
        cap = max(1, args.max_source_len - K)
        if text_ids.size(1) > cap:
            text_ids = text_ids[:, :cap]; text_attn = text_attn[:, :cap]

        emb = model.get_encoder().embed_tokens(text_ids)
        inputs_embeds = torch.cat([prefix, emb], dim=1).to(model_dtype)

        attn_prefix = torch.ones(text_ids.size(0), K, dtype=text_attn.dtype, device=device)
        attention_mask = torch.cat([attn_prefix, text_attn], dim=1)

        # ensure requires_grad True when checkpointing
        if args.grad_checkpoint:
            inputs_embeds.requires_grad_(True)

        return inputs_embeds, attention_mask, labels

    # Configure generation cleanly to avoid deprecation/invalid flag warnings
    def get_generation_config():
        # If sampling requested, force beams=1 and use top-* args
        if (args.top_k and args.top_k > 0) or (args.top_p < 1.0) or (args.temperature != 1.0):
            return GenerationConfig(
                do_sample=True, num_beams=1,
                max_length=args.max_target_len, min_length=args.min_decode_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                top_k=max(1,args.top_k), top_p=args.top_p, temperature=args.temperature,
                early_stopping=True
            )
        else:
            return GenerationConfig(
                do_sample=False, num_beams=args.num_beams,
                max_length=args.max_target_len, min_length=args.min_decode_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                early_stopping=True
            )

    @torch.no_grad()
    def generate_text(inputs_embeds, attention_mask):
        # enforce dtype on inputs at generation time (prevents Float vs BFloat16 mismatch)
        inputs_embeds = inputs_embeds.to(model_dtype)
        gen_cfg = get_generation_config()
        out = model.generate(inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
                             generation_config=gen_cfg)
        return out

    for epoch in range(1,args.epochs+1):
        model.train(); running=0.0
        for step,batch in enumerate(dl_tr,1):
            inputs_embeds, attention_mask, labels = make_inputs(batch)
            # mixed precision optional; model base dtype is float32 — keep stable
            out=model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss=out.loss
            if label_smooth>0: loss=(1.0-label_smooth)*loss

            (loss/args.grad_accum).backward(); step_accum+=1
            if step_accum>=args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step_accum=0
            running+=loss.item()
            if step%100==0:
                print(f"[E{epoch} S{step}] loss={running/step:.4f}")

        # ---- validation ----
        model.eval(); refs,hyps=[] ,[]
        with torch.no_grad():
            for b in dl_va:
                inputs_embeds, attention_mask, labels = make_inputs(b)
                preds_ids = generate_text(inputs_embeds, attention_mask)
                hyps.extend([t.strip() for t in
                             tok.batch_decode(preds_ids, skip_special_tokens=True)])
                refs.extend([t.strip() for t in model_decode(tok, labels)])
        met=compute_metrics(refs,hyps)
        print(f"Epoch {epoch:02d}: train_loss={running/len(dl_tr):.4f}  "
              f"val_ROUGE-L={met['rougeL']:.4f}  val_BLEU4={met['bleu4']:.4f}  val_BERTScoreF1={met['bertscore_f1']:.4f}")

        improved = met["rougeL"]>best_rouge
        if improved:
            best_rouge=met["rougeL"]; no_improve=0
            ckpt={
                "model":model.state_dict(),
                "pooler":pooler.state_dict(),
                "pooler_cfg":{"in_dim":in_dim,"d_model":d_model,
                              "k_tokens":args.k_prefix,"n_heads":args.n_heads,
                              "n_layers":args.n_layers,"dropout":args.dropout},
                "gen_kwargs":{"max_length":args.max_target_len, "min_length":args.min_decode_len,
                              "num_beams":args.num_beams, "no_repeat_ngram_size":args.no_repeat_ngram_size,
                              "length_penalty":args.length_penalty},
                "metrics":met,
                "epoch":epoch,
                "text_model_name":args.text_model_name
            }
            torch.save(ckpt, pathlib.Path(args.out_dir)/"best.ckpt")
            try: tok.save_pretrained(args.out_dir)
            except Exception: pass
            print(f"[Checkpoint] Saved to {args.out_dir} (ROUGE-L={best_rouge:.4f})")
        else:
            no_improve+=1
            if no_improve>=patience:
                print("Early stopping (no validation improvement)."); break
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Best ROUGE-L: {best_rouge:.4f}")

if __name__=="__main__":
    main()
