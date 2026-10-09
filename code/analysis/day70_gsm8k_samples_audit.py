#!/usr/bin/env python3
"""Day70: Day3-proven raw_decode loader + frozen audit_eval_samples.audit()
   for the 3 safe_fallback runs' gsm8k samples (primary metric)."""
import json, sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/campus_week10_11_starter/scripts")
import audit_eval_samples as aes

W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
OUT = Path("/root/autodl-tmp/campus_project/reports/week11")
OUT.mkdir(parents=True, exist_ok=True)

def load_records(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    dec, recs, i, n = json.JSONDecoder(), [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(text, i)
        recs.extend(obj) if isinstance(obj, list) else recs.append(obj)
    return recs

printed_keys = False
for seed in ["20250725", "20250726", "20250727"]:
    run = W11 / f"safe_fallback_seed{seed}"
    files = sorted(run.rglob("samples_gsm8k_*.jsonl"))
    if not files:
        print(f"[MISSING] {run.name} gsm8k samples")
        continue
    f = files[-1]
    recs = load_records(f)
    if not printed_keys and recs:
        print("=== first-record keys (供 bootstrap 解析用) ===")
        for k in recs[0]:
            s = json.dumps(recs[0][k], ensure_ascii=False)
            print(f"  {k}: {s[:150]}")
        printed_keys = True
    summary, details = aes.audit(recs)
    out = OUT / f"safe_fallback_seed{seed}_gsm8k_samples_audit.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / f"safe_fallback_seed{seed}_gsm8k_samples_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {run.name}: n_records={len(recs)} -> {out.name}")
    print("     summary:", json.dumps(summary, ensure_ascii=False)[:300])
