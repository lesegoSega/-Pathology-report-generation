#!/usr/bin/env python3
# Remove boilerplate/signatures/codes and normalize whitespace to densify supervision.

import re, json, argparse, pathlib

JUNK_PATTERNS = [
    r"^Electronically signed.*$", r"^Signature.*$", r"^Charge codes?:.*$",
    r"^Pathologist:.*$", r"^Report (date|time).*$", r"^Specimen (received|submitted).*?$",
]
junk_re = [re.compile(pat, re.I) for pat in JUNK_PATTERNS]

def clean(txt: str) -> str:
    lines = [ln for ln in txt.splitlines() if len(ln.strip())>0]
    keep = []
    for ln in lines:
        if any(r.search(ln) for r in junk_re):
            continue
        keep.append(ln)
    out = " ".join(keep)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    w = open(args.out_jsonl, "w", encoding="utf-8")
    for line in open(args.in_jsonl, "r", encoding="utf-8"):
        if not line.strip(): continue
        obj = json.loads(line)
        rep = obj.get("report","")
        obj["report"] = clean(rep)
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    w.close()
    print("Saved ->", args.out_jsonl)

if __name__ == "__main__":
    main()
