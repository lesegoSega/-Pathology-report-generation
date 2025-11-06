#!/usr/bin/env python3
# Guess organ from report text and write slide_to_organ.csv

import json, argparse, pandas as pd
from train_t5_resampler import _extract_organ_from_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows=[]
    for line in open(args.reports_jsonl,"r",encoding="utf-8"):
        if not line.strip(): continue
        obj=json.loads(line); sid=obj.get("slide_id"); rep=obj.get("report","")
        if not sid: continue
        organ=_extract_organ_from_text(rep) or ""
        rows.append({"slide_id":sid,"organ":organ})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Saved ->", args.out_csv)

if __name__ == "__main__":
    main()
