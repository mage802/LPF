#!/usr/bin/env python3
"""Create disjoint selector and analysis probes from tagged validation data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import read_jsonl, sha256_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-per-skill", type=int, default=8)
    parser.add_argument("--analysis-per-skill", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20250719)
    return parser.parse_args()


def ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["sample_id"]) for row in rows}


def main() -> None:
    args = parse_args()
    tagged = read_jsonl(args.input)
    train_ids = ids(read_jsonl(args.train))
    eligible = [
        row
        for row in tagged
        if float(row.get("skill_label_confidence", 0.0))
        >= args.min_confidence
        and int(row.get("total_tokens", 0)) <= args.max_total_tokens
        and row["sample_id"] not in train_ids
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        groups.setdefault(str(row["primary_skill"]), []).append(row)

    selector_probe, analysis_probe, remainder = [], [], []
    warnings = []
    selected_ids: set[str] = set()
    for skill_index, (skill, rows) in enumerate(sorted(groups.items())):
        rng = random.Random(args.seed + skill_index * 7919)
        ordered = list(rows)
        rng.shuffle(ordered)
        selector_n = min(args.selector_per_skill, max(1, len(ordered) // 3))
        remaining_after_selector = len(ordered) - selector_n
        analysis_n = min(
            args.analysis_per_skill, max(0, remaining_after_selector // 2)
        )
        selector = ordered[:selector_n]
        analysis = ordered[selector_n : selector_n + analysis_n]
        rest = ordered[selector_n + analysis_n :]
        selector_probe.extend(selector)
        analysis_probe.extend(analysis)
        remainder.extend(rest)
        selected_ids.update(ids(selector) | ids(analysis))
        if selector_n < args.selector_per_skill or analysis_n < args.analysis_per_skill:
            warnings.append(
                {
                    "skill": skill,
                    "available": len(ordered),
                    "selector_created": selector_n,
                    "analysis_created": analysis_n,
                }
            )

    ineligible_remainder = [
        row for row in tagged if row["sample_id"] not in selected_ids
    ]
    remainder = ineligible_remainder
    selector_ids = ids(selector_probe)
    analysis_ids = ids(analysis_probe)
    remainder_ids = ids(remainder)
    if selector_ids & analysis_ids:
        raise AssertionError("selector_probe overlaps analysis_probe")
    if selector_ids & train_ids or analysis_ids & train_ids:
        raise AssertionError("probe leakage into train")
    if selector_ids & remainder_ids or analysis_ids & remainder_ids:
        raise AssertionError("probe overlaps remainder")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector_path = args.output_dir / "selector_probe.jsonl"
    analysis_path = args.output_dir / "analysis_probe.jsonl"
    remainder_path = args.output_dir / "validation_remainder.jsonl"
    write_jsonl(selector_path, selector_probe)
    write_jsonl(analysis_path, analysis_probe)
    write_jsonl(remainder_path, remainder)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "train_sha256": sha256_file(args.train),
        "seed": args.seed,
        "constraints": {
            "min_confidence": args.min_confidence,
            "max_total_tokens": args.max_total_tokens,
        },
        "counts": {
            "selector_probe": len(selector_probe),
            "analysis_probe": len(analysis_probe),
            "validation_remainder": len(remainder),
            "selector_by_skill": dict(
                Counter(row["primary_skill"] for row in selector_probe)
            ),
            "analysis_by_skill": dict(
                Counter(row["primary_skill"] for row in analysis_probe)
            ),
        },
        "intersection_checks": {
            "selector_x_analysis": len(selector_ids & analysis_ids),
            "selector_x_train": len(selector_ids & train_ids),
            "analysis_x_train": len(analysis_ids & train_ids),
        },
        "usage_policy": {
            "selector_probe": "scheduler only; never train; not final evaluation",
            "analysis_probe": "offline competence curves and ablations only",
            "final_test": "external GSM8K/HumanEval/IFEval test tasks",
        },
        "warnings": warnings,
        "files": {
            selector_path.name: sha256_file(selector_path),
            analysis_path.name: sha256_file(analysis_path),
            remainder_path.name: sha256_file(remainder_path),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

