#!/usr/bin/env python3
"""Join Week-2 difficulty with Week-3 tags and form balanced skill×difficulty arms."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upstream import ensure_week23_scripts

ensure_week23_scripts()
from common import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged-train", type=Path, required=True)
    parser.add_argument("--scored-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric", default="base_answer_nll")
    parser.add_argument(
        "--bucket-names", nargs="+", default=["easy", "medium", "hard"]
    )
    parser.add_argument("--min-arm-size-warning", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tagged = read_jsonl(args.tagged_train)
    scored = read_jsonl(args.scored_train)
    score_by_id = {row["sample_id"]: row for row in scored}
    if len(score_by_id) != len(scored):
        raise ValueError("Duplicate sample IDs in scored data")
    tagged_ids = {row["sample_id"] for row in tagged}
    missing = tagged_ids - set(score_by_id)
    if missing:
        raise ValueError(f"{len(missing)} tagged samples lack difficulty scores")

    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in tagged:
        score_row = score_by_id[row["sample_id"]]
        if args.metric not in score_row.get("difficulty", {}):
            raise ValueError(f"Missing metric {args.metric}: {row['sample_id']}")
        item = dict(row)
        item["difficulty"] = dict(score_row["difficulty"])
        by_skill.setdefault(str(item["primary_skill"]), []).append(item)

    bucket_count = len(args.bucket_names)
    output_rows = []
    warnings = []
    arm_counts: Counter[str] = Counter()
    for skill, rows in sorted(by_skill.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["difficulty"][args.metric]),
                str(row["sample_id"]),
            ),
        )
        size = len(ordered)
        for rank, row in enumerate(ordered):
            percentile = (rank + 0.5) / size
            bucket_index = min(
                bucket_count - 1, math.floor(percentile * bucket_count)
            )
            bucket = args.bucket_names[bucket_index]
            item = dict(row)
            item["difficulty_metric"] = args.metric
            item["difficulty_value"] = float(
                row["difficulty"][args.metric]
            )
            item["difficulty_percentile_within_skill"] = percentile
            item["difficulty_bucket"] = bucket
            item["difficulty_bucket_center"] = (
                bucket_index + 0.5
            ) / bucket_count
            item["arm_id"] = f"{skill}::{bucket}"
            output_rows.append(item)
            arm_counts[item["arm_id"]] += 1
    for arm, count in sorted(arm_counts.items()):
        if count < args.min_arm_size_warning:
            warnings.append({"arm": arm, "samples": count})

    write_jsonl(args.output, output_rows)
    output_ids = {row["sample_id"] for row in output_rows}
    if output_ids != tagged_ids or len(output_rows) != len(tagged):
        raise AssertionError("Bucket construction changed dataset membership")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tagged_train": str(args.tagged_train.resolve()),
        "tagged_train_sha256": sha256_file(args.tagged_train),
        "scored_train": str(args.scored_train.resolve()),
        "scored_train_sha256": sha256_file(args.scored_train),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "difficulty_metric": args.metric,
        "bucket_names": args.bucket_names,
        "assignment": "equal-frequency quantiles within each primary skill",
        "samples": len(output_rows),
        "skills": len(by_skill),
        "arms": len(arm_counts),
        "arm_counts": dict(sorted(arm_counts.items())),
        "small_arm_warnings": warnings,
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

