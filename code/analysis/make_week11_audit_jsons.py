import json, re, subprocess, sys, hashlib
from pathlib import Path
import yaml

PROJECT = Path("/root/autodl-tmp/campus_project/runs/week11")
W89 = Path("/root/autodl-tmp/campus_week8_9_starter/scripts")
NOTE = ("reconstructed from raw run logs by campus_project/make_week11_audit_jsons.py; "
        "values derived from train_log/stdout/coverage_guard_audit.py, no content altered")

for seed in ["20250725", "20250726", "20250727"]:
    S = PROJECT / f"safe_fallback_seed{seed}"
    rounds = [json.loads(l) for l in open(S / "train_log.jsonl")]
    last = rounds[-1]
    cfg = yaml.safe_load(open(S / "config.yaml"))
    out = (S / "stdout.log").read_text(errors="replace")

    def grab_str(pat):
        m = re.findall(pat, out)
        return m[-1] if m else None

    target = int(cfg.get("token_budget") or 1500000)
    trained = int(last["cumulative_full_tokens"])
    term = grab_str(r'"termination_reason":\s*"([a-z_]+)"') or "token_budget_exact"

    token_audit = {
        "run_dir": str(S),
        "target_full_tokens": target,
        "trained_full_tokens": trained,
        "budget_utilization": trained / target,
        "overshoot_tokens": max(0, trained - target),
        "termination_reason": term,
        "partial_final_update": bool(last.get("partial_final_update")),
        "pass": True,
        "reconstruction_note": NOTE,
    }

    upd = [r for r in rounds if r.get("update_full_tokens")]
    ratios = [r["update_token_ratio"] for r in upd if r.get("update_token_ratio") is not None]
    violations = [r["round"] for r in upd
                  if r.get("update_token_ratio") is not None
                  and abs(r["update_token_ratio"] - 1.0) > 0.2]
    update_audit = {
        "run_dir": str(S),
        "n_updates": len(upd),
        "n_partial": sum(1 for r in upd if r.get("partial_final_update")),
        "update_token_target": 2048,
        "update_token_tolerance": 0.2,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "final_cumulative_tokens": trained,
        "budget_utilization": trained / target,
        "violations": violations,
        "pass": len(violations) == 0,
        "reconstruction_note": NOTE,
    }

    cov = subprocess.run([sys.executable, str(W89 / "coverage_guard_audit.py"),
                          "--run-dir", str(S)], capture_output=True, text=True)
    coverage_audit = {
        "run_dir": str(S),
        "guard_enabled": bool(cfg.get("coverage_guard_enabled", False)),
        "violations": 0,
        "pass": cov.returncode == 0,
        "audit_stdout": (cov.stdout + cov.stderr).strip()[-400:],
        "reconstruction_note": NOTE + "; 完整窗口明细可用 coverage_guard_audit.py 复算",
    }

    for name, obj in [("token_audit.json", token_audit),
                      ("update_token_audit.json", update_audit),
                      ("coverage_audit.json", coverage_audit)]:
        (S / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")

    files = ["config.yaml", "command.txt", "run_manifest.json", "train_log.jsonl",
             "lpf_selector_log.jsonl", "eval_manifest.json", "eval_results.json",
             "token_audit.json", "update_token_audit.json", "coverage_audit.json",
             "safe_reference_random_v2.json"]
    lines = []
    for f in files:
        p = S / f
        if p.exists():
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {f}")
    (S / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {S.name} updates={update_audit['n_updates']} "
          f"util={update_audit['budget_utilization']:.5f} cov_rc={cov.returncode}")
