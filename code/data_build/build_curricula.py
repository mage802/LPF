#!/usr/bin/env python3
"""Build deterministic easy-to-hard and hard-to-easy curriculum views."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, sha256_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["length", "mtld", "base_answer_nll"],
    )
    parser.add_argument("--seed", type=int, default=20250719)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    random_rows = list(rows)
    random.Random(args.seed).shuffle(random_rows)
    random_path = args.output_dir / "random.jsonl"
    write_jsonl(random_path, random_rows)
    files[random_path.name] = sha256_file(random_path)
    for metric in args.metrics:
        if any(metric not in row.get("difficulty", {}) for row in rows):
            raise ValueError(f"Missing difficulty metric: {metric}")
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["difficulty"][metric]),
                str(row["sample_id"]),
            ),
        )
        easy_path = args.output_dir / f"{metric}_easy_to_hard.jsonl"
        hard_path = args.output_dir / f"{metric}_hard_to_easy.jsonl"
        write_jsonl(easy_path, ordered)
        write_jsonl(hard_path, reversed(ordered))
        files[easy_path.name] = sha256_file(easy_path)
        files[hard_path.name] = sha256_file(hard_path)

    source_ids = {row["sample_id"] for row in rows}
    if len(source_ids) != len(rows):
        raise ValueError("Duplicate sample IDs in scored input")
    for path_name in files:
        ids = {row["sample_id"] for row in read_jsonl(args.output_dir / path_name)}
        if ids != source_ids:
            raise AssertionError(f"Curriculum membership drift: {path_name}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "samples": len(rows),
        "metrics": args.metrics,
        "random_seed": args.seed,
        "sort_order": "ascending means easy-to-hard",
        "tie_breaker": "sample_id ascending",
        "files": files,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
