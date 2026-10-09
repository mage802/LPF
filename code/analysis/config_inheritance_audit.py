#!/usr/bin/env python3
"""Check that Week11 Safe-Curriculum changes only preregistered fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allow", default="seed,output_dir,selector_mode,safe_candidate_mode,safe_reference_file,safe_progress_margin,safe_confirm_rounds,safe_fail_cooldown_rounds,coverage_guard_enabled,run_role,schema_version,study_title")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    parent = yaml.safe_load(args.parent.read_text(encoding="utf-8")) or {}
    candidate = yaml.safe_load(args.candidate.read_text(encoding="utf-8")) or {}
    allowed = {item.strip() for item in args.allow.split(",") if item.strip()}
    changed = {}
    for key in sorted(set(parent) | set(candidate)):
        if parent.get(key) != candidate.get(key):
            changed[key] = {"parent": parent.get(key), "candidate": candidate.get(key), "allowed": key in allowed}
    unexpected = [key for key, item in changed.items() if not item["allowed"]]
    payload: dict[str, Any] = {
        "schema_version": "config_inheritance_audit_v1",
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "allowed_fields": sorted(allowed),
        "changed_fields": changed,
        "unexpected_changes": unexpected,
        "audit_status": "fail" if unexpected else "pass",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
