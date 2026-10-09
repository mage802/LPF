#!/usr/bin/env python3
"""Assign deterministic, auditable multi-label skill tags using taxonomy rules."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    read_jsonl,
    sha256_file,
    user_text,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--audit-per-skill", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20250719)
    return parser.parse_args()


def tag_row(
    row: dict[str, Any], taxonomy: dict[str, Any]
) -> dict[str, Any]:
    domain = str(row.get("domain", "general"))
    domain_config = taxonomy["domains"].get(
        domain, taxonomy["domains"]["general"]
    )
    text = f"{user_text(row)}\n{row.get('output', '')}".lower()
    matches = []
    evidence = {}
    for skill in domain_config["skills"]:
        hit_patterns = [
            pattern
            for pattern in skill["patterns"]
            if str(pattern).lower() in text
        ]
        if hit_patterns:
            matches.append(skill["name"])
            evidence[skill["name"]] = hit_patterns
    if matches:
        primary = matches[0]
        confidence = min(0.98, 0.75 + 0.05 * len(evidence[primary]))
        method = "rule_match"
    else:
        primary = domain_config["fallback"]
        matches = [primary]
        evidence[primary] = ["domain_fallback"]
        confidence = 0.55
        method = "domain_fallback"
    result = dict(row)
    result["primary_skill"] = primary
    result["skill_labels"] = matches
    result["skill_label_confidence"] = confidence
    result["skill_label_method"] = method
    result["skill_label_evidence"] = evidence
    result["skill_taxonomy_version"] = taxonomy["version"]
    return result


def main() -> None:
    args = parse_args()
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    tagged = [tag_row(row, taxonomy) for row in rows]
    write_jsonl(args.output, tagged)

    counts = Counter(row["primary_skill"] for row in tagged)
    methods = Counter(row["skill_label_method"] for row in tagged)
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in tagged:
        by_skill.setdefault(row["primary_skill"], []).append(row)
    rng = random.Random(args.seed)
    audit_rows = []
    for skill, candidates in sorted(by_skill.items()):
        chosen = rng.sample(
            candidates, min(args.audit_per_skill, len(candidates))
        )
        for row in chosen:
            audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "domain": row["domain"],
                    "primary_skill": row["primary_skill"],
                    "confidence": row["skill_label_confidence"],
                    "instruction": user_text(row).replace("\n", " ")[:500],
                    "human_correct_0_or_1": "",
                    "human_corrected_skill": "",
                    "human_note": "",
                }
            )
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "taxonomy": str(args.taxonomy.resolve()),
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "taxonomy_version": taxonomy["version"],
        "samples": len(tagged),
        "primary_skill_counts": dict(sorted(counts.items())),
        "label_methods": dict(methods),
        "fallback_ratio": methods["domain_fallback"] / max(1, len(tagged)),
        "manual_audit_rows": len(audit_rows),
        "manual_audit_instruction": (
            "Fill human_correct_0_or_1; target precision >= 0.80 per skill. "
            "Revise taxonomy version instead of silently editing old labels."
        ),
    }
    write_json(args.audit_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

