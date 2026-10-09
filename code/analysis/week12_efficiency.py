import json
from pathlib import Path
W8 = Path("/root/autodl-tmp/campus_week8_9_starter/saves/week8")
W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
OUT = Path("/root/autodl-tmp/week12_results_20260922_033907/tables")
RUNS = [(m, s, W8 / f"{m}_seed202507{s}")
        for m in ("random_sameloop", "lp_only_replay", "lp_match_replay", "lpf_sparse")
        for s in (22, 23, 24)]
RUNS += [("safe_fallback", s, W11 / f"safe_fallback_seed202507{s}") for s in (25, 26, 27)]
keysets, rows = {}, []
for method, seed, d in RUNS:
    m = json.loads((d / "run_manifest.json").read_text())
    t = m.get("timing_seconds", {})
    keysets.setdefault(tuple(sorted(t)), []).append(f"{method}_{seed}")
    wall, src = t.get("training_wall"), "training_wall"
    if wall is None:
        wall = t.get("probe_seconds", 0.0) + t.get("utility_seconds", 0.0) + t.get("train_seconds", 0.0)
        src = "probe+utility+train"
    rows.append({"method": method, "seed": seed, "trained_full_tokens": m.get("trained_full_tokens"),
                 "trained_answer_tokens": m.get("trained_answer_tokens"), "optimizer_steps": m.get("optimizer_steps"),
                 "training_wall_seconds": round(wall, 1), "wall_source": src,
                 "probe_seconds": t.get("probe_seconds"), "utility_seconds": t.get("utility_seconds"),
                 "selector_overhead_ratio": m.get("selector_overhead_ratio")})
print("KEYSETS:")
for ks, runs in keysets.items():
    print(" ", dict.fromkeys(ks), "->", len(runs), "runs")
    if len(ks) < 6: print("    FULL:", dict.fromkeys(ks))
print("SAMPLE timing (first run):", json.dumps(json.loads((RUNS[0][2] / 'run_manifest.json').read_text()).get("timing_seconds", {}), indent=1))
hdr = "| method | seed | full_tok | answer_tok | steps | wall_s (def) | probe_s | utility_s | overhead |"
lines = ["# Week12 efficiency summary: 15 official runs (token_update_matched_v2)", "",
         "wall definition per run recorded in wall_source column; replaces week12_main_table.md wall column (summed-all-keys, double-counted)", "", hdr,
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append("| %s | %d | %s | %s | %s | %.1f (%s) | %s | %s | %.4f |" % (
        r["method"], r["seed"], r["trained_full_tokens"], r["trained_answer_tokens"], r["optimizer_steps"],
        r["training_wall_seconds"], r["wall_source"], round(r["probe_seconds"] or 0, 1),
        round(r["utility_seconds"] or 0, 2), r["selector_overhead_ratio"] or 0))
(OUT / "week12_efficiency_15runs.md").write_text("\n".join(lines) + "\n")
(OUT / "week12_efficiency_15runs.json").write_text(json.dumps({"wall_note": lines[2], "runs": rows}, indent=1))
print("WROTE", OUT / "week12_efficiency_15runs.md")
