import json, glob, random
from pathlib import Path

def load_records(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    dec, recs, i, n = json.JSONDecoder(), [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(text, i)
        recs.append(obj)
    return recs

def strict_scores(path):
    out = {}
    for r in load_records(path):
        if r.get("filter") == "strict-match" and r.get("exact_match") is not None:
            out[int(r["doc_id"])] = 1.0 if float(r["exact_match"]) >= 0.5 else 0.0
    return out

W8 = Path("/root/autodl-tmp/campus_week8_9_starter/saves/week8")
W11 = Path("/root/autodl-tmp/campus_project/runs/week11")

def collect(base, pattern, tag):
    files = sorted(glob.glob(str(base / pattern)))
    assert files, f"no samples: {base}/{pattern}"
    per_seed = []
    for f in files:
        s = strict_scores(f)
        print(f"  [{tag}] {Path(f).parent.parent.parent.name} {Path(f).name}: {len(s)} docs")
        assert len(s) == 1319, f"{f}: strict docs != 1319"
        per_seed.append(s)
    return per_seed

print("[load] safe_fallback:")
safe = collect(W11, "safe_fallback_seed*/eval_artifacts/*/*samples_gsm8k*.jsonl", "safe")
print("[load] random_sameloop:")
rand = collect(W8, "random_sameloop_seed*/eval_artifacts/**/*samples_gsm8k*.jsonl", "rand")
print("[load] lp_match_replay:")
lpm = collect(W8, "lp_match_replay_seed*/eval_artifacts/**/*samples_gsm8k*.jsonl", "lpm")

common = set.intersection(*[set(d) for d in safe + rand + lpm])
docs = sorted(common)
print(f"[pair] common docs: {len(docs)} (expect 1319)")
assert len(docs) == 1319

def method_vec(per_seed):
    return {d: sum(s[d] for s in per_seed) / len(per_seed) for d in docs}

v_safe, v_rand, v_lpm = method_vec(safe), method_vec(rand), method_vec(lpm)
rng = random.Random(20250722)
B = 10000

def boot(a, b, name):
    diffs = [a[d] - b[d] for d in docs]
    delta = sum(diffs) / len(diffs)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    lo, hi = means[int(0.025 * B)], means[int(0.975 * B)]
    return {
        "comparison": name,
        "delta_mean_pp": round(delta * 100, 3),
        "ci95_lower_pp": round(lo * 100, 3),
        "ci95_upper_pp": round(hi * 100, 3),
        "noninferior_delta_ge_-1pp": delta >= -0.01,
        "noninferior_ci_lower_ge_-2.5pp": lo >= -0.025,
        "verdict_noninferior": (delta >= -0.01) and (lo >= -0.025),
    }

res = {
    "schema_version": "example_paired_bootstrap_v1_reconstructed",
    "note": "example-level pairing by doc_id; per-method = mean over 3 seeds; B=10000 percentile CI; rng=20250722; spec from README10-11 section 9",
    "n_common_docs": len(docs),
    "safe_vs_random": boot(v_safe, v_rand, "safe_fallback vs random_sameloop"),
    "safe_vs_lp_match_replay": boot(v_safe, v_lpm, "safe_fallback vs lp_match_replay"),
    "lp_match_replay_vs_random": boot(v_lpm, v_rand, "lp_match_replay vs random_sameloop"),
}
out = Path("/root/autodl-tmp/campus_project/reports/week11/safe_noninferiority_bootstrap.json")
out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
