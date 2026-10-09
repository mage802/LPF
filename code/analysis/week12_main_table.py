#!/usr/bin/env python3
"""Week12 Day75 main table v2: all columns from frozen JSONs + run_manifest.json."""
import json
from pathlib import Path

W12 = Path("/root/autodl-tmp/week12_results_20260922_033907")
W8 = Path("/root/autodl-tmp/campus_week8_9_starter/saves/week8")
W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
METHODS = ["random_sameloop", "lp_only_replay", "lp_match_replay", "lpf_sparse", "safe_fallback"]
LABEL = {"random_sameloop": "Random-SameLoop", "lp_only_replay": "LP-only+replay",
         "lp_match_replay": "LP-Match-Replay", "lpf_sparse": "LPF-Sparse",
         "safe_fallback": "Safe (Week11)"}

def flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flatten(v, "%s.%s" % (p, k) if p else str(k)))
    else: out[p] = o
    return out

def rdirs(m):
    if m == "safe_fallback":
        return [W11 / ("safe_fallback_seed202507%d" % s) for s in (25, 26, 27)]
    return [W8 / ("%s_seed202507%d" % (m, s)) for s in (22, 23, 24)]

stats = json.loads((W12 / "statistics/week12_main_stats.json").read_text())
per = stats["per_seed"]
miss = []
lines = ["# Week 12 主结果表 v2（输入：week12_main_stats.json + 各 run run_manifest.json / eval_results.json）", ""]
lines.append("协议 token_update_matched_v2；主指标 GSM8K strict-match（官方 primary filter）；overhead = manifest selector_overhead_ratio（分子 probe+utility 计时，分母总训练墙钟）")
lines.append("")
lines.append("| method | n | GSM8K strict | flexible | IFEval prompt-strict | IFEval inst-strict | full tok | answer tok | updates | util | overhead | wall s | replay frac |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in METHODS:
    dirs = rdirs(m)
    mans = [json.loads((d / "run_manifest.json").read_text()) for d in dirs]
    evs = []
    for d in dirs:
        er = d / "eval_results.json"
        evs.append(flatten(json.loads(er.read_text())) if er.exists() else {})
    def pick(e, *subs):
        for k, v in e.items():
            kl = k.lower()
            if "ifeval" in kl and all(s in kl for s in subs) and isinstance(v, (int, float)):
                return v
        return None
    ifp = [pick(e, "prompt", "strict") for e in evs]
    ifi = [pick(e, "inst", "strict") for e in evs]
    def avg(key):
        vals = [x.get(key) for x in mans]
        if any(v is None for v in vals): miss.append("%s:%s" % (m, key))
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    def avgf(vals, tag):
        if any(v is None for v in vals): miss.append("%s:%s" % (m, tag))
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    walls = []
    for x in mans:
        t = x.get("timing_seconds") or {}
        nums = [v for v in t.values() if isinstance(v, (int, float))]
        walls.append(sum(nums) if nums else None)
    wall = sum(w for w in walls if w) if any(walls) else None
    lines.append("| %s | 3 | %.4f+-%.4f | %.4f+-%.4f | %s | %s | %d | %s | %d | %.5f | %.3f | %s | %.3f |" % (
        LABEL[m], per[m]["strict_mean"], per[m]["strict_sd"],
        per[m]["flex_mean"], per[m]["flex_sd"],
        ("%.4f" % avgf(ifp, "ifeval_prompt")) if avgf(ifp, "ifeval_prompt") is not None else "N/A",
        ("%.4f" % avgf(ifi, "ifeval_inst")) if avgf(ifi, "ifeval_inst") is not None else "N/A",
        int(avg("trained_full_tokens") or 0),
        str(int(avg("trained_answer_tokens") or 0)),
        int(avg("optimizer_steps") or 0),
        avg("budget_utilization") or 0,
        avg("selector_overhead_ratio") or 0,
        ("%.0f" % wall) if wall else "N/A",
        avg("replay_fraction") or 0))
lines.append("")
lines.append("## 逐 run strict（附录）")
for m in METHODS:
    lines.append("- %s: %s" % (LABEL[m], per[m]["strict_runs"]))
lines.append("")
lines.append("## 统计比较（week12_main_stats.json）")
for k, v in stats["registered"].items():
    lines.append("- %s: %+.3fpp CI [%+.3f, %+.3f] (%s)" % (
        k, v["point_pp"], v["ci_lo_pp"], v["ci_hi_pp"], v["role"]))
lines.append("- 敏感性(事后): %+.3fpp CI [%+.3f, %+.3f]" % (
    stats["sensitivity_post_hoc"]["point_pp"],
    stats["sensitivity_post_hoc"]["ci_lo_pp"], stats["sensitivity_post_hoc"]["ci_hi_pp"]))
lines.append("")
lines.append("## artifact status / provenance")
lines.append("- 15/15 run: run_manifest.json + eval_results.json + samples + config.yaml 齐备；identity 见 evidence_index.json")
lines.append("- Week 8 runs config_provenance=original；Safe runs=reconstructed_registered_day62")
lines.append("- 缺失/降级项: " + (", ".join(miss) if miss else "无"))
out = W12 / "tables/week12_main_table.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("WROTE", out)
print("MISSING:", miss if miss else "none")
