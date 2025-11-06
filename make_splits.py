#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified 90/10 train/val by organ/cancer from reports.jsonl, plus a stratified
test set sampled from annotation712_update.json with slide_ids NOT in reports.jsonl.

Label (organ/cancer) priority per slide_id:
  1) --organ_map_csv  (columns: slide_id,organ)
  2) --organ_terms_csv (columns: term,organ) matched in report text (case-insensitive)
  3) built-in dictionary
  4) 'unknown'

Outputs:
  - splits.json: {"train":[ids...], "val":[ids...], "test":[]}
  - test.jsonl: lines of {"slide_id":..., "report":...} sampled stratified by organ
  - splits_summary.csv: per-split, per-label counts and fractions

Usage example:
  python make_splits.py \
    --reports_jsonl /path/reports.jsonl \
    --annotation_json /path/annotation712_update.json \
    --out_splits /path/splits.json \
    --out_test_json /path/test.jsonl \
    --out_summary_csv /path/splits_summary.csv \
    --seed 13 \
    --train_frac 0.90 \
    --val_frac 0.10 \
    --test_frac_of_reports 0.10 \
    --organ_map_csv /path/slide_to_organ.csv \
    --organ_terms_csv /path/organ_terms.csv
"""

import json
import argparse
from pathlib import Path
import random
import csv
import sys
from typing import List, Tuple, Dict, Any, Optional, DefaultDict
from collections import defaultdict, Counter

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

# -------------------------------
# Normalization helpers
# -------------------------------

def norm_text(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        return " ".join(str(t) for t in x if t is not None).strip()
    return str(x or "").strip()

# -------------------------------
# I/O: reports.jsonl / annotation.json
# -------------------------------

def load_reports_jsonl(path: Path) -> List[Tuple[str, str]]:
    """
    Read reports.jsonl -> list of (slide_id, report).
    Must contain keys: slide_id, report.
    """
    assert path.exists(), f"Not found: {path}"
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = norm_text(obj.get("slide_id"))
            rep = norm_text(obj.get("report"))
            if sid and rep:
                pairs.append((sid, rep))
    return pairs

def iter_records(obj: Any):
    """
    Yield dict-like records from annotation JSON which may be:
      - list[dict]
      - dict[id -> report or list/parts]
      - dict with nested arrays under keys (train/val/test) of dicts
    """
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        looks_like_sets = any(isinstance(v, list) and v and isinstance(v[0], dict) for v in obj.values())
        if looks_like_sets:
            for v in obj.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            yield item
        else:
            for k, v in obj.items():
                yield {"id": k, "report": v}
    else:
        return

def load_annotation_pairs(path: Path) -> List[Tuple[str, str]]:
    """
    Normalize annotation file to list of (id, report).
    Accepts:
      - list of dicts with keys {id or slide_id, report or text}
      - dict of id -> report (report may be string or list)
      - dict with sets {train,val,test}: list[dict{id, report}]
    """
    assert path.exists(), f"Not found: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    total, miss = 0, 0
    for rec in iter_records(data):
        total += 1
        sid = norm_text(rec.get("id") if "id" in rec else rec.get("slide_id"))
        rep_field = rec.get("report", rec.get("text", ""))
        rep = norm_text(rep_field)
        if not sid or not rep:
            miss += 1
            continue
        out.append((sid, rep))
    eprint(f"[ANN] scanned={total} usable={len(out)} skipped={miss}")
    return out

def dedupe_keep_first(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for sid, rep in pairs:
        if sid not in seen:
            seen.add(sid)
            out.append((sid, rep))
    return out

# -------------------------------
# Labeling (organ/cancer)
# -------------------------------

def load_organ_map_csv(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        eprint(f"[WARN] organ_map_csv not found: {path}")
        return {}
    mp: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rid = None
            for k in ("slide_id", "slide", "id"):
                for ck in row.keys():
                    if ck.lower() == k:
                        rid = norm_text(row[ck])
                        break
                if rid:
                    break
            organ = None
            for k in ("organ", "label", "site"):
                for ck in row.keys():
                    if ck.lower() == k:
                        organ = norm_text(row[ck]).lower()
                        break
                if organ:
                    break
            if rid:
                mp[rid] = organ or "unknown"
    return mp

def load_organ_terms_csv(path: Optional[Path]) -> Dict[str, str]:
    """
    Returns mapping term -> organ (lowercased).
    """
    if not path:
        return {}
    if not path.exists():
        eprint(f"[WARN] organ_terms_csv not found: {path}")
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            term = norm_text(row.get("term", "")).lower()
            organ = norm_text(row.get("organ", "")).lower()
            if term and organ:
                out[term] = organ
    return out

def default_organ_terms() -> Dict[str, str]:
    """
    Conservative defaults; extend as needed.
    """
    return {
        # organ terms
        "breast": "breast",
        "mammary": "breast",
        "lung": "lung",
        "pulmonary": "lung",
        "liver": "liver",
        "hepatic": "liver",
        "hepatocellular": "liver",
        "kidney": "kidney",
        "renal": "kidney",
        "colon": "colon",
        "colonic": "colon",
        "rectal": "rectum",
        "rectum": "rectum",
        "prostate": "prostate",
        "thyroid": "thyroid",
        "larynx": "larynx",
        "laryngeal": "larynx",
        "tongue": "tongue",
        "ocular": "eye",
        "eye": "eye",
        "pleura": "pleura",
        "retroperitoneum": "retroperitoneum",
        "skin": "skin",
        "pancreas": "pancreas",
        "pancreatic": "pancreas",
        "gallbladder": "gallbladder",
        "gall bladder": "gallbladder",
        "ovary": "ovary",
        "ovarian": "ovary",
        "maxillary sinus": "maxillary_sinus",
        "sinus": "maxillary_sinus",
        # lymphoma catch
        "lymph node": "lymph_node",
        "lymphoma": "lymph_node",
        # generic cancer keywords to keep as unknown unless paired with organ
        "carcinoma": "unknown",
        "adenocarcinoma": "unknown",
        "squamous": "unknown",
        "melanoma": "unknown",
        "sarcoma": "unknown",
        "glioma": "brain",
        "glioblastoma": "brain",
        "oligodendroglioma": "brain",
        "brain": "brain",
        # thymus
        "thymus": "thymus",
        "thymoma": "thymus",
    }

def infer_label_from_report(report: str,
                            organ_terms_map: Dict[str, str]) -> str:
    txt = (report or "").lower()
    for term in sorted(organ_terms_map.keys(), key=lambda t: (-len(t), t)):
        if term in txt:
            lab = organ_terms_map[term]
            if lab != "unknown":
                return lab
    return "unknown"

def assign_labels(pairs: List[Tuple[str, str]],
                  organ_map: Dict[str, str],
                  organ_terms_map: Dict[str, str]) -> Dict[str, str]:
    """
    Return slide_id -> label
    """
    labels: Dict[str, str] = {}
    for sid, rep in pairs:
        if sid in organ_map and organ_map[sid]:
            labels[sid] = organ_map[sid].lower()
            continue
        labels[sid] = infer_label_from_report(rep, organ_terms_map)
    return labels

# -------------------------------
# Stratified splitting utilities
# -------------------------------

def stratified_split(ids: List[str],
                     labels: Dict[str, str],
                     train_frac: float,
                     rng: random.Random) -> Tuple[List[str], List[str]]:
    """
    Deterministic stratified split by labels dict (slide_id -> label).
    For each label group, take round(train_frac * n_label) to train (min 1 if n>=2).
    """
    by_lab: DefaultDict[str, List[str]] = defaultdict(list)
    for sid in ids:
        lab = labels.get(sid, "unknown")
        by_lab[lab].append(sid)

    train_ids: List[str] = []
    val_ids: List[str] = []

    for lab, sids in by_lab.items():
        rng.shuffle(sids)
        n = len(sids)
        if n == 1:
            n_train = 1 if train_frac >= 0.5 else 0
        else:
            n_train = int(round(train_frac * n))
            n_train = max(1, min(n-1, n_train))  # ensure both sets get at least 1 when n>=2
        train_ids.extend(sids[:n_train])
        val_ids.extend(sids[n_train:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids

def stratified_sample_for_test(candidates: List[Tuple[str, str]],
                               organ_map: Dict[str, str],
                               organ_terms_map: Dict[str, str],
                               target_total: int,
                               ref_label_counts: Counter,
                               rng: random.Random) -> List[Tuple[str, str]]:
    """
    Sample a stratified test set from candidates (id, report), trying to match
    the reference label distribution (from the reports set).
    """
    if target_total <= 0:
        return []

    total_ref = sum(ref_label_counts.values()) or 1
    desired = {lab: int(round((cnt / total_ref) * target_total))
               for lab, cnt in ref_label_counts.items()}

    diff = target_total - sum(desired.values())
    labs_sorted = sorted(ref_label_counts.keys(), key=lambda l: -ref_label_counts[l])
    i = 0
    while diff != 0 and labs_sorted:
        lab = labs_sorted[i % len(labs_sorted)]
        desired[lab] = desired.get(lab, 0) + (1 if diff > 0 else -1 if desired.get(lab, 0) > 0 else 0)
        diff = target_total - sum(desired.values())
        i += 1
        if i > 10000:
            break

    cand_by_lab: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
    for sid, rep in candidates:
        lab = organ_map.get(sid) or infer_label_from_report(rep, organ_terms_map)
        lab = (lab or "unknown").lower()
        cand_by_lab[lab].append((sid, rep))

    chosen: List[Tuple[str, str]] = []
    leftovers: List[Tuple[str, str]] = []

    for lab, want in desired.items():
        pool = cand_by_lab.get(lab, [])
        rng.shuffle(pool)
        take = min(want, len(pool))
        chosen.extend(pool[:take])
        leftovers.extend(pool[take:])

    still_need = target_total - len(chosen)
    if still_need > 0:
        for lab, pool in cand_by_lab.items():
            if lab not in desired:
                leftovers.extend(pool)
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:still_need])

    return chosen[:target_total]

# -------------------------------
# CSV summary
# -------------------------------

def write_splits_summary_csv(path: Path,
                             train_ids: List[str],
                             val_ids: List[str],
                             test_pairs: List[Tuple[str, str]],
                             train_labels: Dict[str, str],
                             val_labels: Dict[str, str],
                             organ_map: Dict[str, str],
                             organ_terms_map: Dict[str, str]) -> None:
    """
    Writes one CSV with rows:
      split,label,count,fraction_of_split,total_in_split,fraction_of_overall,overall_total
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build label lists for each split
    train_labs = [train_labels.get(sid, "unknown") for sid in train_ids]
    val_labs   = [val_labels.get(sid, "unknown")   for sid in val_ids]
    test_labs  = []
    for sid, rep in test_pairs:
        lab = organ_map.get(sid) or infer_label_from_report(rep, organ_terms_map) or "unknown"
        test_labs.append(lab.lower())

    # Counts
    cnt_train = Counter(train_labs)
    cnt_val   = Counter(val_labs)
    cnt_test  = Counter(test_labs)

    # Overall pool for the CSV (train+val+test)
    overall_counts = cnt_train + cnt_val + cnt_test
    overall_total = sum(overall_counts.values()) or 1

    # Collect distinct labels across all splits for stable ordering
    labels_all = sorted(set(list(cnt_train.keys()) + list(cnt_val.keys()) + list(cnt_test.keys())))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "count", "fraction_of_split", "total_in_split",
                    "fraction_of_overall", "overall_total"])

        # Helper to dump one split
        def dump(split_name: str, counts: Counter):
            total_in_split = sum(counts.values()) or 1
            for lab in labels_all:
                c = counts.get(lab, 0)
                frac_split = c / total_in_split if total_in_split else 0.0
                frac_overall = (c / overall_total) if overall_total else 0.0
                w.writerow([split_name, lab, c, f"{frac_split:.6f}", total_in_split,
                            f"{frac_overall:.6f}", overall_total])

        dump("train", cnt_train)
        dump("val",   cnt_val)
        dump("test",  cnt_test)

# -------------------------------
# MAIN
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_jsonl", required=True)
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--out_splits", required=True)
    ap.add_argument("--out_test_json", required=True)
    ap.add_argument("--out_summary_csv", required=True,
                    help="Path to write the per-split per-label summary CSV.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_frac", type=float, default=0.90)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac_of_reports", type=float, default=0.10)
    ap.add_argument("--organ_map_csv", type=str, default=None,
                    help="CSV with columns: slide_id,organ (preferred source for labels)")
    ap.add_argument("--organ_terms_csv", type=str, default=None,
                    help="CSV with columns: term,organ used for keyword→organ mapping")
    args = ap.parse_args()

    if abs(args.train_frac + args.val_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac must equal 1.0")

    rng = random.Random(args.seed)

    # Load label sources
    organ_map = load_organ_map_csv(Path(args.organ_map_csv)) if args.organ_map_csv else {}
    organ_terms_map = load_organ_terms_csv(Path(args.organ_terms_csv)) if args.organ_terms_csv else {}
    if not organ_terms_map:
        organ_terms_map = default_organ_terms()

    # 1) Load & dedupe reports
    rep_pairs = dedupe_keep_first(load_reports_jsonl(Path(args.reports_jsonl)))
    if not rep_pairs:
        raise ValueError("No usable {slide_id, report} pairs found in reports_jsonl")
    rep_ids = [sid for sid, _ in rep_pairs]

    # 2) Assign labels to reports and stratify split
    rep_labels = assign_labels(rep_pairs, organ_map, organ_terms_map)
    train_ids, val_ids = stratified_split(rep_ids, rep_labels, args.train_frac, rng)

    # 3) Build test set (from annotation, excluding reports), stratified to match reports label distribution
    ann_pairs = dedupe_keep_first(load_annotation_pairs(Path(args.annotation_json)))
    rep_id_set = set(rep_ids)
    candidates = [(sid, rep) for (sid, rep) in ann_pairs if sid not in rep_id_set]

    ref_counts = Counter(rep_labels[sid] for sid in rep_ids)

    target_test = max(1, int(round(args.test_frac_of_reports * len(rep_pairs))))
    chosen = stratified_sample_for_test(candidates, organ_map, organ_terms_map, target_test, ref_counts, rng)

    # 4) Write outputs
    out_splits = Path(args.out_splits); out_splits.parent.mkdir(parents=True, exist_ok=True)
    splits_obj: Dict[str, List[str]] = {"train": train_ids, "val": val_ids, "test": []}
    out_splits.write_text(json.dumps(splits_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    out_test = Path(args.out_test_json); out_test.parent.mkdir(parents=True, exist_ok=True)
    with out_test.open("w", encoding="utf-8") as f:
        for sid, rep in chosen:
            f.write(json.dumps({"slide_id": sid, "report": rep}, ensure_ascii=False) + "\n")

    # 5) CSV summary (train/val from rep_labels, test inferred)
    train_labels = {sid: rep_labels.get(sid, "unknown") for sid in train_ids}
    val_labels   = {sid: rep_labels.get(sid, "unknown") for sid in val_ids}
    write_splits_summary_csv(Path(args.out_summary_csv),
                             train_ids, val_ids, chosen,
                             train_labels, val_labels,
                             organ_map, organ_terms_map)

    # 6) Console summary
    def distr(ids: List[str], name: str):
        c = Counter(rep_labels.get(sid, "unknown") for sid in ids)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(6))
        print(f"  {name}: n={len(ids)} | {top}")

    print("----- STRATIFIED SPLIT SUMMARY -----")
    print(f"reports usable pairs: {len(rep_pairs)}")
    distr(train_ids, "train")
    distr(val_ids,   "val")
    print(f"annotation candidates (not in reports): {len(candidates)}")
    print(f"test requested: {target_test}  |  test written: {len(chosen)}")
    test_labels = []
    for sid, rep in chosen:
        lab = organ_map.get(sid) or infer_label_from_report(rep, organ_terms_map) or "unknown"
        test_labels.append(lab.lower())
    print(f"test label distribution: {Counter(test_labels)}")
    print(f"Saved splits      -> {out_splits}")
    print(f"Saved test jsonl  -> {out_test}")
    print(f"Saved summary csv -> {args.out_summary_csv}")

if __name__ == "__main__":
    main()
