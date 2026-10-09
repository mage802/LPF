#!/usr/bin/env python3
"""Minimal safe_fallback patch for train_lpf_v2.py (Week10-11 P1).
Only touches: (1) SafeGate init, (2) selection block gating,
(3) selector log field, (4) utility mode remap. Backs up first.
"""
import shutil, sys, time
from pathlib import Path

TARGET = Path("/root/autodl-tmp/campus_project/code_adapted/week10_11/scripts/train_lpf_v2.py")
src = TARGET.read_text(encoding="utf-8")

if "safe_gate_state" in src:
    sys.exit("already patched, aborting.")

EDITS = []

# --- Edit 1: SafeGate init before the training loop ---
OLD1 = "    timing = Counter()\n    wall_start = time.perf_counter()\n    try:"
NEW1 = '''    timing = Counter()
    wall_start = time.perf_counter()
    safe_gate = None
    if str(config.get("selector_mode", "")) == "safe_fallback":
        from safe_selector import SafeGate

        safe_gate = SafeGate.from_file(
            Path(str(config["safe_reference_file"])).resolve(),
            margin=float(config.get("safe_progress_margin", 0.0)),
            confirm_rounds=int(config.get("safe_confirm_rounds", 2)),
            fail_cooldown_rounds=int(config.get("safe_fail_cooldown_rounds", 2)),
        )
    try:'''
EDITS.append(("init", OLD1, NEW1))

# --- Edit 2: selection block with gate filtering + random fallback ---
OLD2 = '''            explored = rng.random() < float(config["epsilon_exploration"])
            if str(config["selector_mode"]) == "random" or explored:
                pool_sel = sorted(set(arms) & set(coverage_allowed))
                if not pool_sel:
                    pool_sel = list(coverage_allowed)
                selected_arm = rng.choice(pool_sel)
            else:
                allowed = candidate_arms if gradient_updated else list(utilities)
                allowed = [a for a in allowed if a in set(coverage_allowed)]
                if not allowed:
                    allowed = list(coverage_allowed)
                selected_arm = max(
                    allowed,
                    key=lambda arm: utilities[arm]["final_utility"],
                )
            selected_skill = str(arms[selected_arm][0]["primary_skill"])'''
NEW2 = '''            safe_allowed = []
            safe_gate_state = {"enabled": False, "fallback_random": False}
            if safe_gate is not None:
                safe_allowed, safe_gate_state = safe_gate.allowed_arms(
                    arms=arms,
                    skill_state=skill_state,
                    candidate_arms=list(candidate_arms),
                    round_index=round_index,
                )
            safe_blocks = safe_gate is not None and not safe_allowed

            explored = rng.random() < float(config["epsilon_exploration"])
            if str(config["selector_mode"]) == "random" or explored or safe_blocks:
                pool_sel = sorted(set(arms) & set(coverage_allowed))
                if not pool_sel:
                    pool_sel = list(coverage_allowed)
                selected_arm = rng.choice(pool_sel)
            else:
                allowed = candidate_arms if gradient_updated else list(utilities)
                allowed = [a for a in allowed if a in set(coverage_allowed)]
                if safe_gate is not None:
                    allowed = [a for a in allowed if a in set(safe_allowed)]
                if not allowed:
                    allowed = [
                        a for a in list(coverage_allowed)
                        if safe_gate is None or a in set(safe_allowed)
                    ]
                if not allowed:
                    pool_sel = sorted(set(arms) & set(coverage_allowed)) or list(coverage_allowed)
                    selected_arm = rng.choice(pool_sel)
                else:
                    selected_arm = max(
                        allowed,
                        key=lambda arm: utilities[arm]["final_utility"],
                    )
            selected_skill = str(arms[selected_arm][0]["primary_skill"])'''
EDITS.append(("selection", OLD2, NEW2))

# --- Edit 3: log field ---
OLD3 = '                        "coverage_state": coverage_state,\n'
NEW3 = ('                        "coverage_state": coverage_state,\n'
        '                        "safe_gate_state": safe_gate_state,\n')
EDITS.append(("log_field", OLD3, NEW3))

# --- Edit 4: utility mode remap (safe_fallback -> safe_candidate_mode) ---
OLD4 = '    mode = str(config["selector_mode"])\n    result = {}'
NEW4 = '''    mode = str(config["selector_mode"])
    if mode == "safe_fallback":
        mode = str(config.get("safe_candidate_mode", "lp_only"))
    result = {}'''
EDITS.append(("mode_remap", OLD4, NEW4))

for name, old, new in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor '{name}' found {n} times (expected 1); aborting without changes.")
    src = src.replace(old, new, 1)

bak = TARGET.with_suffix(f".py.bak_{time.strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, bak)
TARGET.write_text(src, encoding="utf-8")
print(f"backup : {bak}")
print(f"patched: {TARGET}")
for mk in ("safe_fallback", "SafeGate", "safe_gate_state", "safe_reference_file"):
    print(f"marker '{mk}':", mk in src)
