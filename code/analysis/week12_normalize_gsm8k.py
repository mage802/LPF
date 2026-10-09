#!/usr/bin/env python3
"""Week12 Day73 read-only adapter: 15 runs gsm8k samples -> normalized_gsm8k.jsonl
+ README12 §4.4 four mandatory sample tests + §4.5 quadrants and format diagnostics."""
import json, hashlib
from pathlib import Path
from json import JSONDecoder

W8 = Path("/root/autodl-tmp/campus_week8_9_starter/saves/week8")
W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
W12_OUT = sorted(Path("/root/autodl-tmp").glob("week12_results_*"))[-1] / "sample_checks"
W12_OUT.mkdir(parents=True, exist_ok=True)

RUNS = []
for m in ["random_sameloop", "lp_only_replay", "lp_match_replay", "lpf_sparse"]:
    for s in (22, 23, 24):
        RUNS.append(("%s_seed202507%d" % (m, s), m, 20250700 + s, W8 / ("%s_seed202507%d" % (m, s))))
for s in (25, 26, 27):
    RUNS.append(("safe_fallback_seed202507%d" % s, "safe_fallback", 20250700 + s,
                 W11 / ("safe_fallback_seed202507%d" % s)))

def load_records(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    dec, recs, i, n = JSONDecoder(), [], 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(text, i)
        recs.extend(obj if isinstance(obj, list) else [obj])
    return recs

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    else:
        out[prefix] = obj
    return out

report, normalized, all_doc_hashes = {}, [], {}
raw_mismatch_total = 0

for run_id, method, seed, rdir in RUNS:
    entry = {"run_dir": str(rdir)}
    files = sorted(rdir.rglob("samples_gsm8k_*.jsonl"))
    if len(files) != 1:
        entry["error"] = "samples files found: %s" % [str(f) for f in files]
        report[run_id] = entry
        print("[ERROR] %s: %s" % (run_id, entry["error"]))
        continue
    f = files[0]
    entry["samples_file"] = str(f)
    entry["samples_sha256"] = sha256(f)
    recs = load_records(f)
    entry["n_records"] = len(recs)
    assert len(recs) == 2638, "%s: n_records=%d != 2638" % (run_id, len(recs))

    by_doc = {}
    for r in recs:
        assert r.get("filter") in ("strict-match", "flexible-extract"), r.get("filter")
        by_doc.setdefault(str(r["doc_id"]), {})[r["filter"]] = r

    raw_mismatch = 0
    rows = []
    quad = {"11": 0, "01": 0, "00": 0, "10": 0}
    diag = {"empty": 0, "marker_missing": 0, "long_no_marker": 0}
    for d, pair in by_doc.items():
        assert "strict-match" in pair and "flexible-extract" in pair, "%s doc %s missing filter" % (run_id, d)
        rs, rf = pair["strict-match"], pair["flexible-extract"]
        raw_s = (rs.get("resps") or [[None]])[0][0]
        raw_f = (rf.get("resps") or [[None]])[0][0]
        if raw_s != raw_f:
            raw_mismatch += 1
        strict = int(bool(rs.get("exact_match")))
        flexible = int(bool(rf.get("exact_match")))
        quad["%d%d" % (strict, flexible)] += 1
        raw = raw_s if isinstance(raw_s, str) else None
        if raw is None or raw.strip() == "":
            diag["empty"] += 1
        else:
            if "####" not in raw:
                diag["marker_missing"] += 1
                if len(raw) > 800:
                    diag["long_no_marker"] += 1
        rows.append({"run_id": run_id, "method": method, "seed": seed,
                     "task": "gsm8k", "doc_id": d,
                     "doc_hash": rs.get("doc_hash"),
                     "strict": strict, "flexible": flexible,
                     "raw_response": raw,
                     "raw_response_available": raw is not None,
                     "source_file_sha256": entry["samples_sha256"]})
    raw_mismatch_total += raw_mismatch
    entry["n_docs"] = len(rows)
    entry["raw_mismatch_across_filters"] = raw_mismatch
    entry["quadrant"] = quad
    entry["diag"] = diag
    entry["strict_mean"] = sum(r["strict"] for r in rows) / len(rows)
    entry["flexible_mean"] = sum(r["flexible"] for r in rows) / len(rows)
    all_doc_hashes[run_id] = {r["doc_id"]: r["doc_hash"] for r in rows}

    er = rdir / "eval_results.json"
    entry["official"] = {}
    if er.exists():
        flat = flatten(json.loads(er.read_text(encoding="utf-8")))
        cands = {k: v for k, v in flat.items()
                 if "gsm8k" in k and isinstance(v, (int, float))
                 and k.rsplit(".", 1)[-1].startswith("exact_match")}
        entry["official"]["candidates"] = cands
        TOL = 5e-5
        entry["official"]["strict_match_key"] = next(
            (k for k, v in cands.items() if abs(v - entry["strict_mean"]) <= TOL), None)
        entry["official"]["flexible_match_key"] = next(
            (k for k, v in cands.items() if abs(v - entry["flexible_mean"]) <= TOL), None)
    else:
        entry["official"]["error"] = "eval_results.json not found"

    entry["tests"] = {
        "t1_unique_key": len({(r["run_id"], r["doc_id"]) for r in rows}) == len(rows) == 1319,
        "t2_raw_equal": raw_mismatch == 0,
        "t3_official_strict": entry["official"].get("strict_match_key") is not None,
        "t3_official_flexible": entry["official"].get("flexible_match_key") is not None,
    }
    report[run_id] = entry
    normalized.extend(rows)
    print("[OK] %s docs=%d strict=%.4f flexible=%.4f quad=%s strict_key=%s" % (
        run_id, len(rows), entry["strict_mean"], entry["flexible_mean"], quad,
        entry["official"].get("strict_match_key")))

ref_run = next(iter(all_doc_hashes))
ref_set = set(all_doc_hashes[ref_run].values())
t4 = {r: (set(all_doc_hashes[r].values()) == ref_set) for r in all_doc_hashes}
print("t4 doc_hash consistent:", t4)
print("raw_mismatch_total:", raw_mismatch_total)

(W12_OUT / "normalized_gsm8k.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in normalized) + "\n",
    encoding="utf-8")
(W12_OUT / "normalized_tests.json").write_text(json.dumps(
    {"runs": report, "t4_doc_hash_consistent": t4,
     "raw_mismatch_total": raw_mismatch_total}, indent=2, ensure_ascii=False),
    encoding="utf-8")
(W12_OUT / "quadrant_counts.json").write_text(json.dumps(
    {r: report[r].get("quadrant") for r in report}, indent=2), encoding="utf-8")
print("WROTE", W12_OUT / "normalized_gsm8k.jsonl")
print("WROTE", W12_OUT / "normalized_tests.json")
print("WROTE", W12_OUT / "quadrant_counts.json")
