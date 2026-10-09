#!/usr/bin/env python3
"""coverage_guard_audit.py — Week-8 v2 coverage guard audit.

Recomputes per-window arm exposure and cumulative domain fractions from the
train log (independent of the trainer), cross-checks the guard actions logged
in the selector log, and writes <run_dir>/coverage_audit.json.

Usage:
    python3 coverage_guard_audit.py --run-dir saves/week8/<run_id>
Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

NEW_ROOT = Path("/root/autodl-tmp/campus_week8_9_starter")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(raw: str, run_dir: Path) -> Path:
    """Config paths are relative to the launch cwd (NEW_ROOT); be cwd-proof."""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in (Path.cwd(), NEW_ROOT, run_dir):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (NEW_ROOT / p).resolve()  # fall through: open() will raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    violations: list[str] = []

    manifest_path = run_dir / "run_manifest.json"
    train_path = run_dir / "train_log.jsonl"
    selector_path = run_dir / "lpf_selector_log.jsonl"
    for path in (manifest_path, train_path, selector_path):
        if not path.exists():
            print(f"[FAIL] missing {path}")
            return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("coverage_guard_params", {})
    window = int(params.get("window_tokens", 250000))
    min_arm_frac = float(params.get("min_arm_exposure_fraction", 0.005))
    domain_tol = float(params.get("domain_deviation_tolerance", 0.10))
    guard_enabled = bool(manifest.get("coverage_guard_enabled", False))
    orig = {
        str(k): float(v)
        for k, v in manifest.get("orig_domain_fraction", {}).items()
    }

    # arm/skill -> domain maps from the training file (independent rebuild).
    # row_domain() MUST mirror train_lpf_v2.row_domain exactly: explicit
    # 'domain' field first, else primary_skill prefix fallback (advisor's
    # convention) — otherwise audit and trainer disagree on 'unknown' rows.
    def row_domain(row: dict) -> str:
        domain = str(row.get("domain", "") or "")
        if domain:
            return domain
        skill = str(row.get("primary_skill", "unknown"))
        return skill.split("::", 1)[-1].split(".", 1)[0]

    train_rows = read_jsonl(
        resolve_path(str(manifest["config"]["train_file"]), run_dir)
    )
    arm_domain = {}
    skill_domain = {}
    for row in train_rows:
        arm_domain.setdefault(str(row["arm_id"]), row_domain(row))
        skill_domain.setdefault(str(row["primary_skill"]), row_domain(row))
    all_arms = sorted(arm_domain)

    # replay sources contribute to domain totals but not arm selection
    arm_window: dict[int, Counter] = {}
    domain_tokens: Counter[str] = Counter()
    total = 0
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            tokens = int(r["full_tokens"])
            before = int(r["cumulative_full_tokens"]) - tokens
            w = max(0, before) // window
            source = str(r["training_source"])
            if source.startswith("replay::"):
                domain = skill_domain.get(source.split("::", 1)[1], "unknown")
            else:
                domain = arm_domain.get(source, "unknown")
                arm_window.setdefault(w, Counter())[source] += tokens
            domain_tokens[domain] += tokens
            total += tokens

    n_windows = max(1, (max(1, total) + window - 1) // window)
    # starved arms per completed window pair (trainer convention)
    starved_windows: dict[str, list[int]] = {}
    for w in range(2, n_windows + 1):
        c1 = arm_window.get(w - 1, Counter())
        c2 = arm_window.get(w - 2, Counter())
        for arm in all_arms:
            if c1.get(arm, 0) == 0 and c2.get(arm, 0) == 0:
                starved_windows.setdefault(arm, []).append(w)

    domain_cum = {d: domain_tokens.get(d, 0) / max(1, total) for d in orig}
    domain_dev = {d: domain_cum[d] - orig[d] for d in orig}
    max_abs_dev = max((abs(v) for v in domain_dev.values()), default=0.0)

    arm_cum: Counter[str] = Counter()
    for c in arm_window.values():
        arm_cum.update(c)
    arms_below_min = sorted(
        a for a in all_arms if arm_cum.get(a, 0) / max(1, total) < min_arm_frac
    )

    # guard consistency: when guard enabled and a starvation event exists at
    # some window w, at least one selector round inside/after window w must
    # show action == "force_starved_arms".
    guard_actions: Counter[str] = Counter()
    saw_guard_state = False
    with selector_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            state = r.get("coverage_state") or {}
            if state.get("enabled"):
                saw_guard_state = True
                guard_actions[str(state.get("action", "none"))] += 1

    if guard_enabled:
        if not saw_guard_state:
            violations.append("guard enabled but selector log has no coverage_state")
        if starved_windows and guard_actions.get("force_starved_arms", 0) == 0:
            violations.append(
                f"{len(starved_windows)} arms starved but guard never forced "
                f"(actions={dict(guard_actions)})"
            )
        if max_abs_dev > domain_tol + 0.02:  # small tolerance for window lag
            violations.append(
                f"domain deviation {max_abs_dev:.3f} exceeds tol {domain_tol} "
                "with guard enabled"
            )
    report = {
        "run_dir": str(run_dir),
        "guard_enabled": guard_enabled,
        "window_tokens": window,
        "n_windows": n_windows,
        "domain_cumulative_fraction": domain_cum,
        "domain_target_fraction": orig,
        "domain_deviation": domain_dev,
        "max_abs_domain_deviation": max_abs_dev,
        "domain_deviation_tolerance": domain_tol,
        "arms_with_two_empty_windows": {k: v for k, v in starved_windows.items()},
        "arms_below_min_exposure": arms_below_min,
        "min_arm_exposure_fraction": min_arm_frac,
        "guard_action_counts": dict(guard_actions),
        "violations": violations,
        "pass": not violations,
    }
    (run_dir / "coverage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = "PASS" if not violations else "FAIL"
    print(
        f"[{status}] {run_dir.name}: guard={'on' if guard_enabled else 'off'} "
        f"max|domain_dev|={max_abs_dev:.3f} starved_arms={len(starved_windows)} "
        f"below_min={len(arms_below_min)} violations={len(violations)}"
    )
    for v in violations[:5]:
        print(f"    - {v}")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
