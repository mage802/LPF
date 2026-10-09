#!/usr/bin/env python3
"""Week12 Day74: per-seed table + registered bootstrap recompute + post-hoc seed sensitivity.
Reads sample_checks/normalized_gsm8k.jsonl (Day73 output). Writes statistics/."""
import json
from pathlib import Path
import numpy as np

W12_OUT = sorted(Path("/root/autodl-tmp").glob("week12_results_*"))[-1]
SC = W12_OUT / "sample_checks"
ST = W12_OUT / "statistics"
ST.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in (SC / "normalized_gsm8k.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
docs = sorted({r["doc_id"] for r in rows})
n_docs = len(docs)
assert n_docs == 1319, n_docs
methods = {}
for r in rows:
    methods.setdefault(r["method"], {}).setdefault(r["run_id"], {})[r["doc_id"]] = r
for m in methods:
    assert len(methods[m]) == 3, (m, len(methods[m]))
    for rid in methods[m]:
        assert len(methods[m][rid]) == 1319, (rid, len(methods[m][rid]))

doc_idx = {d: i for i, d in enumerate(docs)}
def strict_mat(m):
    M = np.zeros((3, n_docs))
    for j, rid in enumerate(sorted(methods[m])):
        for d, r in methods[m][rid].items():
            M[j, doc_idx[d]] = r["strict"]
    return M

M = {m: strict_mat(m) for m in methods}
seed_ids = {m: sorted(methods[m]) for m in methods}

print("=== per-seed strict / flexible ===")
per_seed = {}
for m in sorted(methods):
    fs = {m2: None for m2 in []}
    st = [methods[m][rid][d]["strict"] for rid in seed_ids[m] for d in docs]
    fl = [methods[m][rid][d]["flexible"] for rid in seed_ids[m] for d in docs]
    pass
lines = ["| method | run | strict | flexible |", "|---|---|---:|---:|"]
for m in sorted(methods):
    ss, ff = [], []
    for rid in seed_ids[m]:
        s = [methods[m][rid][d]["strict"] for d in docs]
        f = [methods[m][rid][d]["flexible"] for d in docs]
        ss.append(np.mean(s)); ff.append(np.mean(f))
        lines.append("| %s | %s | %.4f | %.4f |" % (m, rid, ss[-1], ff[-1]))
    lines.append("| **%s mean+-sd (n=3)** | - | **%.4f+-%.4f** | **%.4f+-%.4f** |" % (
        m, np.mean(ss), np.std(ss, ddof=1), np.mean(ff), np.std(ff, ddof=1)))
    per_seed[m] = {"strict_runs": [round(x, 4) for x in ss],
                   "strict_mean": round(float(np.mean(ss)), 4),
                   "strict_sd": round(float(np.std(ss, ddof=1)), 4),
                   "flex_mean": round(float(np.mean(ff)), 4),
                   "flex_sd": round(float(np.std(ff, ddof=1)), 4), "n": 3}
    print("%-18s strict %s mean=%.4f sd=%.4f" % (m, [round(x, 4) for x in ss], np.mean(ss), np.std(ss, ddof=1)))
(ST / "week12_per_seed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def registered_boot(A, Bm, rng):
    sA = M[A].mean(axis=0); sB = M[Bm].mean(axis=0)
    d = sA - sB
    point = float(d.mean())
    idx = rng.integers(0, n_docs, size=(10000, n_docs))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"A": A, "B": Bm, "point_pp": round(point * 100, 3),
            "ci_lo_pp": round(float(lo) * 100, 3), "ci_hi_pp": round(float(hi) * 100, 3),
            "B": 10000, "unit": "example_level_shared_doc_resampling",
            "rng": "numpy.default_rng(20250722)"}

rng = np.random.default_rng(20250722)
pairs = [("safe_fallback", "random_sameloop", "primary_preregistered"),
         ("lp_match_replay", "random_sameloop", "reproduction_of_registered"),
         ("safe_fallback", "lp_match_replay", "secondary_not_equivalence"),
         ("safe_fallback", "lp_only_replay", "exploratory_vs_actual_parent")]
results = {}
for A, Bm, label in pairs:
    res = registered_boot(A, Bm, rng)
    res["role"] = label
    if label == "primary_preregistered":
        res["noninferiority_rule"] = "pass iff point>=-1.00pp and ci_lo>=-2.50pp"
        res["verdict"] = bool(res["point_pp"] >= -1.00 and res["ci_lo_pp"] >= -2.50)
    results["%s_minus_%s" % (A, Bm)] = res
    print("%s - %s: %+.3fpp CI [%+.3f, %+.3f] (%s)%s" % (
        A, Bm, res["point_pp"], res["ci_lo_pp"], res["ci_hi_pp"], label,
        " verdict=%s" % res.get("verdict") if "verdict" in res else ""))

print("=== post-hoc seed-resampling sensitivity (safe vs random), rng 20250723 ===")
rng2 = np.random.default_rng(20250723)
A, Bm = "safe_fallback", "random_sameloop"
diffs = np.empty(10000)
for b in range(10000):
    iA = rng2.integers(0, 3, 3); iB = rng2.integers(0, 3, 3)
    dd = rng2.integers(0, n_docs, n_docs)
    diffs[b] = M[A][iA][:, dd].mean() - M[Bm][iB][:, dd].mean()
lo2, hi2 = np.percentile(diffs, [2.5, 97.5])
sens = {"comparison": "safe_minus_random", "post_hoc": True,
        "point_pp": round(float(diffs.mean()) * 100, 3),
        "ci_lo_pp": round(float(lo2) * 100, 3), "ci_hi_pp": round(float(hi2) * 100, 3),
        "algorithm": "resample 3 run indices within each group independently + shared doc indices",
        "rng": "numpy.default_rng(20250723)", "B": 10000,
        "conditionality": "conditional on frozen reference and the 3+3 trained runs; not algorithm-level coverage"}
print("sensitivity: %+.3fpp CI [%+.3f, %+.3f]" % (sens["point_pp"], sens["ci_lo_pp"], sens["ci_hi_pp"]))

out = {"per_seed": per_seed, "registered": results, "sensitivity_post_hoc": sens,
       "input_file": str(SC / "normalized_gsm8k.jsonl"),
       "note": "primary verdict uses registered margins; sensitivity is post-hoc and reported side-by-side, not a replacement"}
(ST / "week12_main_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("WROTE", ST / "week12_main_stats.json")
print("WROTE", ST / "week12_per_seed.md")
