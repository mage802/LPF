#!/usr/bin/env python3
"""Frozen Random-fallback safety gate for the Week 10-11 confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SafeGate:
    """Stateful, deterministic per-skill eligibility gate.

    The gate only filters candidate arms. It never changes utility weights.
    When it returns no arm, the trainer must use the Random fallback.
    """

    def __init__(self, reference: dict[str, Any], *, margin: float = 0.0,
                 confirm_rounds: int = 2, fail_cooldown_rounds: int = 2) -> None:
        if confirm_rounds < 1 or fail_cooldown_rounds < 0:
            raise ValueError("invalid SafeGate confirmation/cooldown values")
        self.reference = reference
        self.reference_by_skill = reference.get("reference_by_skill", {})
        self.margin = float(margin)
        self.confirm_rounds = int(confirm_rounds)
        self.fail_cooldown_rounds = int(fail_cooldown_rounds)
        self.confirm_counts: dict[str, int] = {}
        self.active: dict[str, bool] = {}
        self.cooldown_until: dict[str, int] = {}

    @classmethod
    def from_file(cls, path: Path, **kwargs: Any) -> "SafeGate":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "safe_reference_v1":
            raise ValueError("safe reference must use schema_version=safe_reference_v1")
        return cls(data, **kwargs)

    @staticmethod
    def _skill(arms: dict[str, list[dict[str, Any]]], arm: str) -> str:
        return str((arms.get(arm) or [{}])[0].get("primary_skill", arm))

    def _lower(self, skill: str) -> float | None:
        item = self.reference_by_skill.get(skill, {})
        value = item.get("lower_bound", item.get("mean_learning_progress"))
        return float(value) if value is not None else None

    def allowed_arms(self, *, arms: dict[str, list[dict[str, Any]]],
                     skill_state: dict[str, dict[str, Any]],
                     candidate_arms: list[str], round_index: int) -> tuple[list[str], dict[str, Any]]:
        skills = sorted({self._skill(arms, arm) for arm in candidate_arms})
        detail: dict[str, Any] = {}
        active_skills: list[str] = []
        eligible_skills: list[str] = []
        for skill in skills:
            observed = skill_state.get(skill, {}).get("learning_progress")
            lower = self._lower(skill)
            cooldown = self.cooldown_until.get(skill, -1)
            qualifies = (lower is not None and observed is not None
                         and float(observed) >= lower + self.margin
                         and round_index >= cooldown)
            self.confirm_counts[skill] = self.confirm_counts.get(skill, 0) + 1 if qualifies else 0
            if not qualifies and self.active.get(skill, False):
                self.active[skill] = False
                self.cooldown_until[skill] = round_index + self.fail_cooldown_rounds
            if self.confirm_counts.get(skill, 0) >= self.confirm_rounds:
                self.active[skill] = True
            if self.active.get(skill, False) and round_index >= self.cooldown_until.get(skill, -1):
                active_skills.append(skill)
            if qualifies:
                eligible_skills.append(skill)
            detail[skill] = {
                "observed_learning_progress": observed,
                "reference_lower_bound": lower,
                "qualifies_this_round": qualifies,
                "confirm_count": self.confirm_counts.get(skill, 0),
                "active": self.active.get(skill, False),
                "cooldown_until": self.cooldown_until.get(skill, -1),
            }
        allowed = [arm for arm in candidate_arms if self._skill(arms, arm) in active_skills]
        return allowed, {
            "enabled": True,
            "round": round_index,
            "active_skills": active_skills,
            "eligible_skills": eligible_skills,
            "per_skill": detail,
            "fallback_random": not bool(allowed),
            "reference_schema_version": self.reference.get("schema_version"),
        }


__all__ = ["SafeGate"]
