#!/usr/bin/env python3
"""Measure answer-only PPL for every skill in a fixed probe set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import (
    aggregate_nll,
    load_model_and_tokenizer,
    per_example_answer_nll,
    read_jsonl,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter-path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.probe)
    model, tokenizer, device, dtype = load_model_and_tokenizer(
        args.model_name,
        revision=args.model_revision,
        adapter_path=args.adapter_path,
    )
    values = per_example_answer_nll(
        model,
        tokenizer,
        rows,
        args.max_length,
        args.batch_size,
        device,
    )
    skill_by_id = {
        row["sample_id"]: row["primary_skill"] for row in rows
    }
    enriched = [
        {**row, "primary_skill": skill_by_id[row["sample_id"]]}
        for row in values
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe": str(args.probe.resolve()),
        "probe_sha256": sha256_file(args.probe),
        "model_name": args.model_name,
        "model_revision_requested": args.model_revision,
        "model_revision_loaded": getattr(model.config, "_commit_hash", None),
        "adapter_path": args.adapter_path,
        "device": str(device),
        "dtype": str(dtype),
        "loss_scope": "assistant tokens only",
        "skills": aggregate_nll(enriched, "primary_skill"),
        "examples": enriched,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

