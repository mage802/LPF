#!/usr/bin/env python3
"""Generate W12_OUT/evidence_index.json for the 15 official runs."""
import json, hashlib
from pathlib import Path

W8 = Path("/root/autodl-tmp/campus_week8_9_starter/saves/week8")
W11 = Path("/root/autodl-tmp/campus_project/runs/week11")
W12_OUT = sorted(Path("/root/autodl-tmp").glob("week12_results_*"))[-1]

ROLES = {"random_sameloop": "primary_baseline", "lp_only_replay": "parent_policy_expected",
         "lp_match_replay": "frozen_failed_candidate_v1", "lpf_sparse": "sparse_probe_control",
         "safe_fallback": "confirmation_candidate"}
PROV = {"safe_fallback": "reconstructed_registered_day62", }

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

runs = []
spec = []
for m in ["random_sameloop", "lp_only_replay", "lp_match_replay", "lpf_sparse"]:
    for s in (22, 23, 24):
        spec.append((m, 20250700 + s, W8 / ("%s_seed%d" % (m, 20250700 + s))))
for s in (25, 26, 27):
    spec.append(("safe_fallback", 20250700 + s, W11 / ("safe_fallback_seed%d" % (20250700 + s))))

for m, seed, rdir in spec:
    rid = rdir.name
    e = {"run_id": rid, "run_dir": str(rdir), "method": m, "seed": seed,
         "protocol_family": "token_update_matched_v2", "role": ROLES[m]}
    mf = rdir / "run_manifest.json"
    e["manifest_sha256"] = sha256(mf) if mf.exists() else None
    man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    e["selector_mode"] = man.get("selector_mode") or man.get("config", {}).get("selector_mode")
    cfgs = sorted(rdir.glob("*config*.yaml")) + sorted(rdir.glob("*.yaml"))
    e["effective_config_path"] = str(cfgs[0]) if cfgs else None
    e["config_provenance"] = PROV.get(m, "original")
    e["method_identity_evidence"] = "selector_mode + effective config + run_manifest sha256 + trainer/safe_selector source sha (see server_schema_notes)"
    g = sorted(rdir.rglob("samples_gsm8k_*.jsonl"))
    i = sorted(rdir.rglob("samples_ifeval_*.jsonl"))
    e["samples_path_gsm8k"] = str(g[0]) if g else None
    e["samples_path_ifeval"] = str(i[0]) if i else None
    runs.append(e)
    print(rid, "| sel_mode:", e["selector_mode"], "| cfg:", e["effective_config_path"],
          "| gsm8k:", bool(g), "| ifeval:", bool(i))

out = {"schema_version": "week12_evidence_index_v1", "runs": runs}
p = W12_OUT / "evidence_index.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
json.loads(p.read_text(encoding="utf-8"))
print("WROTE", p, "n_runs:", len(runs))
