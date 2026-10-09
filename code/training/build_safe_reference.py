#!/usr/bin/env python3
"""Build a frozen per-skill Random-SameLoop learning-progress reference."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True,
                        help="Random-SameLoop run directory; repeat for seeds")
    parser.add_argument("--burn-in-rounds", type=int, default=1)
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    values: dict[str, list[float]] = defaultdict(list)
    used_runs: list[str] = []
    for run_dir in args.run_dir:
        path = run_dir / "lpf_selector_log.jsonl"
        if not path.exists():
            path = run_dir / "selector_log.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"selector log missing: {run_dir}")
        rows = read_jsonl(path)
        used_runs.append(str(run_dir))
        for row in rows:
            if int(row.get("round", 0)) < args.burn_in_rounds:
                continue
            state = row.get("skill_state", {})
            if not isinstance(state, dict):
                continue
            for skill, item in state.items():
                if isinstance(item, dict) and item.get("learning_progress") is not None:
                    values[str(skill)].append(float(item["learning_progress"]))
    reference: dict[str, dict[str, Any]] = {}
    for skill, series in sorted(values.items()):
        if not series:
            continue
        mean = statistics.mean(series)
        sd = statistics.stdev(series) if len(series) > 1 else 0.0
        se = sd / math.sqrt(len(series)) if series else 0.0
        reference[skill] = {
            "n": len(series),
            "mean_learning_progress": mean,
            "sd_learning_progress": sd,
            "standard_error": se,
            "lower_bound": mean - args.z_value * se,
        }
    if not reference:
        raise ValueError("no skill learning_progress values found")
    payload = {
        "schema_version": "safe_reference_v1",
        "source_method": "random_sameloop",
        "run_dirs": used_runs,
        "burn_in_rounds": args.burn_in_rounds,
        "z_value": args.z_value,
        "reference_by_skill": reference,
        "frozen_rule": "candidate skill must clear lower_bound + margin for confirm_rounds consecutive probes; otherwise Random fallback",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
