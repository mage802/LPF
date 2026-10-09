#!/usr/bin/env python3
"""Day72 compact: full JSON -> gate_checks/, one screen line per run."""
import json
from pathlib import Path
from json import JSONDecoder

W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
W12_OUT = sorted(Path("/root/autodl-tmp").glob("week12_results_*"))[-1] / "gate_checks"
W12_OUT.mkdir(parents=True, exist_ok=True)
RUNS = [("seed25", W11 / "safe_fallback_seed20250725"),
        ("seed26", W11 / "safe_fallback_seed20250726"),
        ("seed27", W11 / "safe_fallback_seed20250727"),
        ("smoke", Path("/root/autodl-tmp/campus_project/runs/week11_smoke/safe_fallback_seed20250725"))]
REF = json.load(open("/root/autodl-tmp/campus_project/reports/week10/safe_reference_random_v2.json"))
LOW = {k: v["lower_bound"] for k, v in REF["reference_by_skill"].items()}

def load(path):
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

full = {}
for name, rdir in RUNS:
    recs = load(rdir / "lpf_selector_log.jsonl")
    n = len(recs)
    skills = sorted(recs[0]["safe_gate_state"]["per_skill"].keys())
    per = {s: {"pass": 0, "qual": 0, "act": 0, "vals": []} for s in skills}
    fb = fb_while_active = sel_outside = 0
    elig_s, act_s = [], []
    for r in recs:
        st = r["safe_gate_state"]
        if st.get("fallback_random") is True:
            fb += 1
        elig, act = st.get("eligible_skills", []), st.get("active_skills", [])
        elig_s.append(len(elig))
        act_s.append(len(act))
        if st.get("fallback_random") and act:
            fb_while_active += 1
        sel = r.get("selected_arm")
        if sel and not st.get("fallback_random") and sel.split("::")[0] not in act:
            sel_outside += 1
        for s, v in st.get("per_skill", {}).items():
            lp = v.get("observed_learning_progress")
            if lp is not None:
                per[s]["vals"].append(lp)
                if s in LOW and lp >= LOW[s]:
                    per[s]["pass"] += 1
            per[s]["qual"] += bool(v.get("qualifies_this_round"))
            per[s]["act"] += bool(v.get("active"))
    med = lambda xs: sorted(xs)[len(xs) // 2]
    ps = {}
    for s in skills:
        vv = per[s]["vals"]
        ps[s] = {"pass%": round(100 * per[s]["pass"] / n, 1),
                 "qual%": round(100 * per[s]["qual"] / n, 1),
                 "active%": round(100 * per[s]["act"] / n, 1),
                 "lp_mean": round(sum(vv) / len(vv), 4) if vv else None,
                 "lp_min": round(min(vv), 4) if vv else None,
                 "lp_max": round(max(vv), 4) if vv else None,
                 "lower_bound": LOW.get(s)}
    full[name] = {"n_rounds": n, "fallback_true": fb, "fallback_fraction": round(fb / n, 4),
                  "fallback_while_active_nonempty": fb_while_active,
                  "selected_outside_active_when_not_fallback": sel_outside,
                  "eligible_med": med(elig_s), "active_med": med(act_s),
                  "active_min": min(act_s), "active_max": max(act_s),
                  "rounds_zero_active": sum(1 for a in act_s if a == 0),
                  "skills_without_probe": [s for s in skills if not per[s]["vals"]],
                  "per_skill": ps}
    print("== %s rounds=%d fb=%d/%d zero_active=%d fb_while_active=%d sel_outside=%d elig_med=%d act_med=%d noprobe=%s" % (
        name, n, fb, n, full[name]["rounds_zero_active"], fb_while_active, sel_outside,
        full[name]["eligible_med"], full[name]["active_med"], full[name]["skills_without_probe"]))

out = W12_OUT / "gate_compact_summary.json"
out.write_text(json.dumps(full, indent=2), encoding="utf-8")
print("WROTE", out)
