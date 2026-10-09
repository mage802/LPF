#!/usr/bin/env python3
"""Replay Week-3 probe logs to inspect LPF state before any new GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upstream import ensure_week23_scripts

ensure_week23_scripts()
from common import write_jsonl  # noqa: E402
from train_lpf import update_skill_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema-fast-decay", type=float, default=0.50)
    parser.add_argument("--ema-slow-decay", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = []
    for line in args.selector_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            source_rows.append(json.loads(line))
    state = {}
    output_rows = []
    for index, row in enumerate(source_rows):
        scores = row.get("probe_scores")
        if not scores:
            continue
        round_index = int(row.get("round", index))
        state = update_skill_state(
            state,
            scores,
            args.ema_fast_decay,
            args.ema_slow_decay,
            round_index,
        )
        output_rows.append(
            {
                "round": round_index,
                "source_selected_skill": row.get("selected_skill"),
                "skill_state": state,
            }
        )
    write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "source_rounds": len(source_rows),
                "replayed_probe_updates": len(output_rows),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

