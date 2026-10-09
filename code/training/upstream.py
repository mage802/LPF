#!/usr/bin/env python3
"""Locate the completed Week-2/3 scripts without hard-coding a home path."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_week23_scripts() -> Path:
    candidates = []
    configured = os.environ.get("CAMPUS_WEEK23_SCRIPTS")
    if configured:
        candidates.append(Path(configured))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent,
            here.parents[2] / "campus_week2_3_starter" / "scripts",
            Path.cwd().parent / "campus_week2_3_starter" / "scripts",
        ]
    )
    for candidate in candidates:
        if (candidate / "common.py").exists() and (
            candidate / "train_simplified_campus.py"
        ).exists():
            resolved = candidate.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    raise FileNotFoundError(
        "Cannot find Week-2/3 scripts. Set CAMPUS_WEEK23_SCRIPTS to the "
        "absolute path of campus_week2_3_starter/scripts."
    )

