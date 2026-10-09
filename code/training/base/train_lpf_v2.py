#!/usr/bin/env python3
"""Train LPF curriculum under the Week-8 token_update_matched_v2 protocol.

Self-contained v2 trainer. Depends only on the stable Week2-5 modules
(common, gradient_sketch, upstream); it does NOT import train_simplified_campus,
so drift in the v1 scripts cannot leak into v2 runs.

v2 additions on top of the Week6-7 token_matched_v1 trainer:
  1. update_token_target / update_token_tolerance: every optimizer update is
     planned to consume ~update_token_target full tokens (within tolerance),
     never crossing the remaining global budget.
  2. token_normalized_gradient: each planned micro-batch loss is weighted by
     its answer-token share, so one update equals the per-answer-token mean
     gradient over the whole update.
  3. token-progress cosine LR: lr is a function of seen_full_tokens /
     target_full_tokens (no step-count estimate).
  4. replay hysteresis (on/off thresholds + cooldown + hard fraction cap).
  5. coverage guard: restricts the CANDIDATE ARM SET ONLY; it never touches
     utility weights (protocol requirement §2.2-5).

The script refuses to run unless config.protocol_family == "token_update_matched_v2"
and strict_token_budget is true, so a v1 config can never be mislabeled as v2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW

# Lives in campus_week8_9_starter/scripts/; make the Week4-5 shared modules
# (upstream/common/gradient_sketch) importable regardless of cwd.
_WEEK45_SCRIPTS = Path("/root/autodl-tmp/campus_week4_5_starter/scripts")
try:
    if _WEEK45_SCRIPTS.is_dir() and str(_WEEK45_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_WEEK45_SCRIPTS))
except OSError:
    pass  # not on the server / no access — import error below will say so

from upstream import ensure_week23_scripts  # noqa: E402  (Week4-5 module)

ensure_week23_scripts()
from common import (  # noqa: E402
    aggregate_nll,
    collate_supervised,
    load_model_and_tokenizer,
    per_example_answer_nll,
    read_jsonl,
    seed_everything,
    sha256_file,
    tokenize_supervised,
    write_json,
)
from gradient_sketch import cosine, gradient_sketch  # noqa: E402

REQUIRED_PROTOCOL = "token_update_matched_v2"


def row_domain(row: dict[str, Any]) -> str:
    """Domain of a training row: explicit 'domain' field first, else the
    frozen skill-name prefix (advisor convention, e.g. 'math.add' -> 'math').
    The coverage audit script mirrors this exact fallback order.
    """
    domain = str(row.get("domain", "") or "")
    if domain:
        return domain
    return str(row.get("primary_skill", "unknown")).split("::", 1)[-1].split(".", 1)[0]


# ---------------------------------------------------------------------------
# v2: token-progress cosine LR
# ---------------------------------------------------------------------------
class TokenProgressCosine:
    """LR schedule driven by consumed full tokens, not optimizer steps.

    lr = base * t / warmup_tokens                       (warmup)
    lr = base * 0.5 * (1 + cos(pi * progress))          (after warmup)
    where progress is measured in tokens beyond the warmup phase.
    """

    def __init__(
        self,
        optimizer: Any,
        base_lr: float,
        warmup_tokens: int,
        total_tokens: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_tokens = max(1, int(warmup_tokens))
        self.total_tokens = max(self.warmup_tokens + 1, int(total_tokens))
        self._lr = 0.0
        self.set_seen_tokens(0)

    def set_seen_tokens(self, seen: int) -> None:
        seen = max(0, int(seen))
        if seen < self.warmup_tokens:
            lr = self.base_lr * seen / self.warmup_tokens
        else:
            span = self.total_tokens - self.warmup_tokens
            progress = min(1.0, (seen - self.warmup_tokens) / span)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        self._lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:  # compatibility no-op; progress drives lr
        return None

    def get_last_lr(self) -> list[float]:
        return [self._lr]


# ---------------------------------------------------------------------------
# v2: update-token planning + token-normalized gradient
# ---------------------------------------------------------------------------
def plan_micro_batch(
    pool: list[dict[str, Any]],
    rng: random.Random,
    tokenizer: Any,
    max_length: int,
    micro_batch_size: int,
    remaining_tokens: int,
) -> list[dict[str, Any]]:
    """Tokenize candidate rows and keep only what fits the token remainder.

    Falls back to the single shortest fitting row in the pool so a long
    random candidate can never stall or overshoot the budget.
    """
    candidate_rows = choose_rows(pool, micro_batch_size, rng)
    candidate_items = [
        item
        for item in (
            tokenize_supervised(row, tokenizer, max_length)
            for row in candidate_rows
        )
        if item["answer_tokens"] > 0
    ]
    fitting: list[dict[str, Any]] = []
    used = 0
    for item in candidate_items:
        item_tokens = int(item["total_tokens"])
        if used + item_tokens <= remaining_tokens:
            fitting.append(item)
            used += item_tokens
    if not fitting:
        fitting = [
            item
            for item in (
                tokenize_supervised(row, tokenizer, max_length)
                for row in pool
            )
            if item["answer_tokens"] > 0
            and int(item["total_tokens"]) <= remaining_tokens
        ]
        if fitting:
            fitting = [min(fitting, key=lambda item: int(item["total_tokens"]))]
    return fitting


def micro_loss_weights(
    answer_counts: list[int], token_normalized: bool = True
) -> list[float]:
    """Per-micro-batch loss weights.

    token_normalized=True -> weight = answer_tokens_i / total_answer_tokens,
    i.e. the update gradient equals the per-answer-token mean gradient.
    False -> uniform 1/N (legacy mean-of-means), kept for unit tests only.
    """
    n = len(answer_counts)
    if n == 0:
        return []
    if not token_normalized:
        return [1.0 / n] * n
    total = max(1, sum(int(a) for a in answer_counts))
    return [int(a) / total for a in answer_counts]


def plan_update_batches(
    pool: list[dict[str, Any]],
    rng: random.Random,
    tokenizer: Any,
    max_length: int,
    micro_batch_size: int,
    update_token_target: int,
    update_token_tolerance: float,
    global_remaining_tokens: int,
) -> tuple[list[list[dict[str, Any]]], int, int]:
    """Plan one update's micro-batches before any forward pass.

    cap   = min(target * (1 + tol), global_remaining)
    floor = min(target * (1 - tol), global_remaining)
    Micro-batches are appended until planned tokens reach the floor, never
    exceeding the cap. Returns (planned_batches, floor, cap).
    """
    target = int(update_token_target)
    tol = float(update_token_tolerance)
    global_remaining = max(0, int(global_remaining_tokens))
    cap = int(min(target * (1.0 + tol), global_remaining))
    floor = int(min(target * (1.0 - tol), global_remaining))
    planned: list[list[dict[str, Any]]] = []
    planned_tokens = 0
    while planned_tokens < floor:
        remaining = cap - planned_tokens
        if remaining <= 0:
            break
        items = plan_micro_batch(
            pool, rng, tokenizer, max_length, micro_batch_size, remaining
        )
        if not items:
            break
        planned.append(items)
        planned_tokens += sum(int(i["total_tokens"]) for i in items)
    return planned, floor, cap


def train_optimizer_step_v2(
    model: Any,
    tokenizer: Any,
    pool: list[dict[str, Any]],
    rng: random.Random,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
    micro_batch_size: int,
    max_grad_norm: float,
    update_token_target: int,
    update_token_tolerance: float,
    global_remaining_tokens: int,
    token_normalized_gradient: bool = True,
    seen_full_tokens: int = 0,
    prefixed_batches: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """One optimizer update under the v2 protocol.

    Planning happens entirely before the forward pass (see
    plan_update_batches). If the global budget is smaller than the floor the
    update is marked partial_final_update. An update that cannot place a
    single token returns full_tokens=0 (budget_exhausted) so the caller can
    stop with termination_reason=token_budget_exact.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if hasattr(scheduler, "set_seen_tokens"):
        scheduler.set_seen_tokens(seen_full_tokens)

    target = int(update_token_target)
    tol = float(update_token_tolerance)
    if prefixed_batches is not None:
        planned = prefixed_batches
        planned_tokens = sum(
            int(item["total_tokens"]) for items in planned for item in items
        )
        global_remaining = max(0, int(global_remaining_tokens))
        cap = int(min(target * (1.0 + tol), global_remaining))
        floor = int(min(target * (1.0 - tol), global_remaining))
    else:
        planned, floor, cap = plan_update_batches(
            pool,
            rng,
            tokenizer,
            max_length,
            micro_batch_size,
            update_token_target,
            update_token_tolerance,
            global_remaining_tokens,
        )
        planned_tokens = sum(
            int(item["total_tokens"]) for items in planned for item in items
        )

    if not planned:
        return {
            "mean_micro_loss": 0.0,
            "full_tokens": 0,
            "answer_tokens": 0,
            "sample_ids": [],
            "learning_rate": scheduler.get_last_lr()[0],
            "budget_exhausted": 1,
            "partial_final_update": 0,
            "update_full_tokens": 0,
            "target_update_full_tokens": target,
            "update_token_ratio": 0.0,
            "token_normalized_gradient": int(token_normalized_gradient),
            "n_micro_batches_planned": 0,
        }

    answer_per_micro = [
        sum(int(item["answer_tokens"]) for item in items) for items in planned
    ]
    weights = micro_loss_weights(answer_per_micro, token_normalized_gradient)

    effective_loss = 0.0
    full_tokens = 0
    answer_tokens = 0
    sample_ids: list[str] = []
    for items, weight in zip(planned, weights):
        batch = collate_supervised(items, tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            loss = model(**batch).loss
            scaled = loss * float(weight)
        scaled.backward()
        effective_loss += float(loss.detach().item()) * float(weight)
        full_tokens += sum(int(item["total_tokens"]) for item in items)
        answer_tokens += sum(int(item["answer_tokens"]) for item in items)
        sample_ids.extend(str(item["sample_id"]) for item in items)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    scheduler.step()

    ratio = full_tokens / max(1, target)
    return {
        "mean_micro_loss": effective_loss,
        "full_tokens": full_tokens,
        "answer_tokens": answer_tokens,
        "sample_ids": sample_ids,
        "learning_rate": scheduler.get_last_lr()[0],
        "budget_exhausted": 0,
        "partial_final_update": int(full_tokens < floor or planned_tokens < target * (1.0 - tol)),
        "update_full_tokens": full_tokens,
        "target_update_full_tokens": target,
        "update_token_ratio": ratio,
        "token_normalized_gradient": int(token_normalized_gradient),
        "n_micro_batches_planned": len(planned),
    }


# ---------------------------------------------------------------------------
# v2: replay hysteresis (on/off thresholds + cooldown + hard fraction cap)
# ---------------------------------------------------------------------------
class ReplayHysteresis:
    """State machine: off -> on (forgetting >= on_threshold),
    on -> cooldown (forgetting < off_threshold), cooldown -> off after N rounds.
    A hard cap forces replay inactive whenever the cumulative replay token
    fraction reaches max_replay_fraction. All transitions are logged.
    """

    def __init__(
        self,
        on_threshold: float,
        off_threshold: float,
        cooldown_rounds: int,
        max_replay_fraction: float,
    ) -> None:
        self.on = float(on_threshold)
        self.off = float(off_threshold)
        self.cooldown_rounds = int(cooldown_rounds)
        self.max_fraction = float(max_replay_fraction)
        self.state = "off"
        self.cooldown_left = 0
        self.transitions: list[dict[str, Any]] = []

    def update(
        self, forgetting: float, round_index: int, current_replay_fraction: float
    ) -> tuple[bool, dict[str, Any]]:
        action = "hold"
        if self.state == "on" and forgetting < self.off:
            self.state = "cooldown"
            self.cooldown_left = self.cooldown_rounds
            action = "on->cooldown"
        elif self.state == "cooldown":
            self.cooldown_left -= 1
            if self.cooldown_left <= 0:
                self.state = "off"
                action = "cooldown->off"
        elif self.state == "off" and forgetting >= self.on:
            self.state = "on"
            action = "off->on"
        capped = bool(
            self.state == "on" and current_replay_fraction >= self.max_fraction
        )
        active = self.state == "on" and not capped
        info = {
            "state": self.state,
            "cooldown_left": self.cooldown_left,
            "forgetting": float(forgetting),
            "capped": capped,
            "on_threshold": self.on,
            "off_threshold": self.off,
            "max_replay_fraction": self.max_fraction,
            "transition": action,
        }
        if action != "hold":
            self.transitions.append(
                {
                    "round": int(round_index),
                    "transition": action,
                    "forgetting": float(forgetting),
                }
            )
        return active, info


# ---------------------------------------------------------------------------
# v2: coverage guard (candidate-set restriction ONLY; never touches utilities)
# ---------------------------------------------------------------------------
class CoverageGuard:
    """Per-window skill exposure + cumulative domain-deviation guard.

    Semantics (fixed a priori, recorded in the manifest):
      - A window covers `window_tokens` full tokens; tokens of an update that
        crosses a boundary are attributed to the window in which it STARTED.
      - An arm is "starved" if it received zero SELECTION tokens in both of
        the two most recently completed windows. Replay tokens do not count
        toward arm selection exposure (they bypass arm selection) but DO
        count toward domain token totals.
      - Domain deviation compares cumulative domain token fractions with the
        original training-set fractions (measured in tokens, not rows).
      - Enforcement only restricts the allowed arm set, in priority order:
        starved arms > under-exposed domains > exclude over-exposed domains.
        Utility weights are never modified (protocol requirement §2.2-5).
    """

    def __init__(
        self,
        arm_domain: dict[str, str],
        orig_domain_fraction: dict[str, float],
        window_tokens: int,
        min_arm_exposure_fraction: float,
        domain_deviation_tolerance: float,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.arm_domain = {str(a): str(d) for a, d in arm_domain.items()}
        self.orig = {str(d): float(f) for d, f in orig_domain_fraction.items()}
        self.window = max(1, int(window_tokens))
        self.min_arm_fraction = float(min_arm_exposure_fraction)
        self.domain_tolerance = float(domain_deviation_tolerance)
        self.arm_tokens_by_window: dict[int, Counter[str]] = {}
        self.domain_tokens: Counter[str] = Counter()
        self.total_tokens = 0

    def observe(
        self,
        arm: str | None,
        domain: str,
        tokens: int,
        seen_tokens_before: int,
    ) -> None:
        window = int(max(0, seen_tokens_before)) // self.window
        bucket = self.arm_tokens_by_window.setdefault(window, Counter())
        if arm is not None:
            bucket[str(arm)] += int(tokens)
        self.domain_tokens[str(domain)] += int(tokens)
        self.total_tokens += int(tokens)

    def cumulative_arm_tokens(self) -> Counter[str]:
        merged: Counter[str] = Counter()
        for counter in self.arm_tokens_by_window.values():
            merged.update(counter)
        return merged

    def allowed_arms(
        self, all_arms: list[str], seen_tokens: int
    ) -> tuple[list[str], dict[str, Any]]:
        all_arms = sorted(str(a) for a in all_arms)
        if not self.enabled:
            return list(all_arms), {"enabled": False}
        current_window = int(max(0, seen_tokens)) // self.window
        starved: list[str] = []
        if current_window >= 2:
            c1 = self.arm_tokens_by_window.get(current_window - 1, Counter())
            c2 = self.arm_tokens_by_window.get(current_window - 2, Counter())
            starved = [
                arm for arm in all_arms if c1.get(arm, 0) == 0 and c2.get(arm, 0) == 0
            ]
        cumulative_domain = {
            d: self.domain_tokens.get(d, 0) / max(1, self.total_tokens)
            for d in self.orig
        }
        over = sorted(
            d
            for d in self.orig
            if cumulative_domain[d] > self.orig[d] + self.domain_tolerance
        )
        under = sorted(
            d
            for d in self.orig
            if cumulative_domain[d] < self.orig[d] - self.domain_tolerance
        )
        if starved:
            allowed, action = starved, "force_starved_arms"
        elif under:
            allowed = [a for a in all_arms if self.arm_domain.get(a) in under]
            action = "force_underexposed_domain"
            if not allowed:
                allowed, action = list(all_arms), "none"
        elif over:
            allowed = [a for a in all_arms if self.arm_domain.get(a) not in over]
            action = "exclude_overexposed_domain"
            if not allowed:
                allowed, action = list(all_arms), "none"
        else:
            allowed, action = list(all_arms), "none"
        arm_cum = self.cumulative_arm_tokens()
        arms_below_min = sorted(
            a
            for a in all_arms
            if arm_cum.get(a, 0) / max(1, self.total_tokens) < self.min_arm_fraction
        )
        state = {
            "enabled": True,
            "window": int(current_window),
            "starved_arms": starved,
            "over_domains": over,
            "under_domains": under,
            "domain_cumulative_fraction": cumulative_domain,
            "domain_target_fraction": self.orig,
            "arms_below_min_exposure": arms_below_min,
            "action": action,
            "allowed_arm_count": len(allowed),
        }
        return sorted(allowed), state


# ---------------------------------------------------------------------------
# Week4-7 selector logic (copied verbatim from the Week6-7 trainer)
# ---------------------------------------------------------------------------
def probe_scores(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    values = per_example_answer_nll(
        model, tokenizer, rows, max_length, batch_size, device
    )
    skill_by_id = {
        row["sample_id"]: str(row["primary_skill"]) for row in rows
    }
    enriched = [
        {**value, "primary_skill": skill_by_id[value["sample_id"]]}
        for value in values
    ]
    return aggregate_nll(enriched, "primary_skill")


def update_skill_state(
    previous: dict[str, dict[str, float]],
    scores: dict[str, dict[str, float | int]],
    fast_decay: float,
    slow_decay: float,
    round_index: int,
) -> dict[str, dict[str, float]]:
    state = {}
    for skill, score in sorted(scores.items()):
        current = float(score["mean_nll"])
        if skill not in previous:
            fast = slow = best = initial = current
            fast_delta = 0.0
        else:
            old = previous[skill]
            fast = fast_decay * old["ema_fast_nll"] + (
                1.0 - fast_decay
            ) * current
            slow = slow_decay * old["ema_slow_nll"] + (
                1.0 - slow_decay
            ) * current
            best = min(old["best_ema_fast_nll"], fast)
            initial = old["initial_nll"]
            fast_delta = old["ema_fast_nll"] - fast
        progress = slow - fast
        forgetting = max(0.0, (fast - best) / max(best, 1e-8))
        relative_competence = float(
            np.clip(1.0 - fast / max(initial, 1e-8), 0.0, 1.0)
        )
        state[skill] = {
            "observed_nll": current,
            "ema_fast_nll": fast,
            "ema_slow_nll": slow,
            "best_ema_fast_nll": best,
            "initial_nll": initial,
            "learning_progress": progress,
            "one_update_progress": fast_delta,
            "forgetting": forgetting,
            "competence_raw": math.exp(-min(20.0, fast)),
            "competence_relative": relative_competence,
            "last_probe_round": float(round_index),
        }
    return state


def zscores(values: dict[str, float]) -> dict[str, float]:
    array = np.array(list(values.values()), dtype=np.float64)
    mean = float(array.mean())
    std = max(float(array.std()), 1e-8)
    return {key: (value - mean) / std for key, value in values.items()}


def base_arm_utilities(
    arms: dict[str, list[dict[str, Any]]],
    state: dict[str, dict[str, float]],
    last_selected_round: dict[str, int],
    round_index: int,
    config: dict[str, Any],
    progress_fraction: float = 0.0,
) -> dict[str, dict[str, float]]:
    loss_z = zscores(
        {skill: values["ema_fast_nll"] for skill, values in state.items()}
    )
    progress_z = zscores(
        {
            skill: values["learning_progress"]
            for skill, values in state.items()
        }
    )
    forgetting_z = zscores(
        {skill: values["forgetting"] for skill, values in state.items()}
    )
    max_stale = max(
        1,
        max(
            round_index - last_selected_round.get(arm, -1)
            for arm in arms
        ),
    )
    mode = str(config["selector_mode"])
    result = {}
    for arm, rows in arms.items():
        skill = str(rows[0]["primary_skill"])
        bucket_center = float(rows[0]["difficulty_bucket_center"])
        target = float(
            np.clip(
                state.get(skill, {}).get("competence_relative", 0.0)
                + float(config["competence_margin"]),
                0.0,
                1.0,
            )
        )
        match = -abs(bucket_center - target)
        staleness = (
            round_index - last_selected_round.get(arm, -1)
        ) / max_stale
        if mode in ("forward_static", "backward_static"):
            static_target = progress_fraction if mode == "forward_static" else 1.0 - progress_fraction
            target = static_target  # log fields below must show the schedule, not competence
            match = -abs(bucket_center - static_target)
            utility = (
                float(config["difficulty_match_weight"]) * match
                + float(config["staleness_weight"]) * staleness
            )
        elif mode == "random":
            utility = 0.0
        elif mode == "loss_only":
            utility = loss_z.get(skill, 0.0) + float(
                config["staleness_weight"]
            ) * staleness
        elif mode == "lp_only":
            utility = progress_z.get(skill, 0.0) + float(
                config["staleness_weight"]
            ) * staleness
        else:
            utility = (
                float(config["learning_progress_weight"])
                * progress_z.get(skill, 0.0)
                + float(config["staleness_weight"]) * staleness
            )
            if mode != "no_forgetting":
                utility += (
                    float(config["forgetting_weight"])
                    * forgetting_z.get(skill, 0.0)
                )
            if mode != "no_match":
                utility += (
                    float(config["difficulty_match_weight"]) * match
                )
        result[arm] = {
            "base_utility": utility,
            "final_utility": utility,
            "skill_loss_z": loss_z.get(skill, 0.0),
            "learning_progress_z": progress_z.get(skill, 0.0),
            "forgetting_z": forgetting_z.get(skill, 0.0),
            "difficulty_match": match,
            "target_difficulty": target,
            "bucket_center": bucket_center,
            "staleness": staleness,
            "gradient_cosine": 0.0,
            "gradient_conflict": 0.0,
        }
    return result


def choose_rows(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if len(rows) >= count:
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def apply_gradient_conflict(
    model: Any,
    tokenizer: Any,
    arms: dict[str, list[dict[str, Any]]],
    utilities: dict[str, dict[str, float]],
    reference_sketch: torch.Tensor | None,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    round_index: int,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, torch.Tensor],
    list[str],
    dict[str, Any],
]:
    top_k = min(int(config["gradient_candidate_top_k"]), len(utilities))
    candidates = sorted(
        utilities,
        key=lambda arm: utilities[arm]["base_utility"],
        reverse=True,
    )[:top_k]
    sketches: dict[str, torch.Tensor] = {}
    stats = {}
    for index, arm in enumerate(candidates):
        skill = str(arms[arm][0]["primary_skill"])
        if skill in sketches:
            continue
        rng = random.Random(seed + round_index * 1009 + index * 97)
        selected_rows = choose_rows(
            arms[arm], int(config["gradient_examples_per_skill"]), rng
        )
        sketch, gradient_stats = gradient_sketch(
            model,
            tokenizer,
            selected_rows,
            int(config["max_length"]),
            device,
            int(config["gradient_sketch_dim"]),
            int(config["gradient_coords_per_tensor"]),
            seed + round_index * 31 + index,
        )
        sketches[skill] = sketch
        stats[skill] = {
            **gradient_stats,
            "arm_used": arm,
            "sample_ids": [row["sample_id"] for row in selected_rows],
        }
    for arm in candidates:
        skill = str(arms[arm][0]["primary_skill"])
        similarity = (
            cosine(sketches[skill], reference_sketch)
            if reference_sketch is not None
            else 0.0
        )
        conflict = max(0.0, -similarity)
        utilities[arm]["gradient_cosine"] = similarity
        utilities[arm]["gradient_conflict"] = conflict
        utilities[arm]["final_utility"] -= (
            float(config["gradient_conflict_weight"]) * conflict
        )
    return utilities, sketches, candidates, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--selector-mode",
        choices=[
            "lpf",
            "loss_only",
            "lp_only",
            "no_forgetting",
            "no_match",
            "random",
            "forward_static",
            "backward_static",
        ],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument(
        "--replay-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--gradient-conflict-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--probe-interval-rounds", type=int)
    parser.add_argument("--gradient-interval-rounds", type=int)
    return parser.parse_args()


def _row_full_tokens(row: dict[str, Any]) -> int | None:
    """Full-token count from pre-computed fields, or None if absent."""
    if "total_tokens" in row:
        return int(row["total_tokens"])
    if "prompt_tokens" in row and "answer_tokens" in row:
        return int(row["prompt_tokens"]) + int(row["answer_tokens"])
    return None


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.selector_mode:
        config["selector_mode"] = args.selector_mode
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_optimizer_steps is not None:
        config["max_optimizer_steps"] = args.max_optimizer_steps
    if args.token_budget is not None:
        config["token_budget"] = args.token_budget
    if args.replay_enabled is not None:
        config["replay_enabled"] = args.replay_enabled
    if args.gradient_conflict_enabled is not None:
        config["gradient_conflict_enabled"] = args.gradient_conflict_enabled
    if args.probe_interval_rounds:
        config["probe_interval_rounds"] = args.probe_interval_rounds
    if args.gradient_interval_rounds:
        config["gradient_interval_rounds"] = args.gradient_interval_rounds

    # --- protocol gate: refuse to mislabel anything as v2 ------------------
    if str(config.get("protocol_family")) != REQUIRED_PROTOCOL:
        sys.exit(
            f"[v2] REFUSE: config.protocol_family="
            f"{config.get('protocol_family')!r} != {REQUIRED_PROTOCOL!r}. "
            "This trainer only runs the Week-8 v2 protocol."
        )
    if not bool(config.get("strict_token_budget")):
        sys.exit("[v2] REFUSE: strict_token_budget must be true in v2.")
    update_token_target = int(config.get("update_token_target", 0))
    if update_token_target <= 0:
        sys.exit("[v2] REFUSE: update_token_target must be a positive int.")
    update_token_tolerance = float(config.get("update_token_tolerance", 0.20))
    token_normalized_gradient = bool(config.get("token_normalized_gradient", True))

    seed = int(config["seed"])
    seed_everything(seed)
    rng = random.Random(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(config["train_file"])
    probe_rows = read_jsonl(config["selector_probe_file"])
    train_ids = {row["sample_id"] for row in train_rows}
    probe_ids = {row["sample_id"] for row in probe_rows}
    if train_ids & probe_ids:
        raise ValueError("selector probe leaked into LPF training data")

    arms: dict[str, list[dict[str, Any]]] = {}
    by_skill: dict[str, list[dict[str, Any]]] = {}
    arm_domain: dict[str, str] = {}
    skill_domain: dict[str, str] = {}
    for row in train_rows:
        arm_id = str(row["arm_id"])
        skill = str(row["primary_skill"])
        domain = row_domain(row)
        arms.setdefault(arm_id, []).append(row)
        by_skill.setdefault(skill, []).append(row)
        arm_domain.setdefault(arm_id, domain)
        skill_domain.setdefault(skill, domain)
    if set(arm_domain.values()) == {"unknown"}:
        print("[v2] WARNING: cannot resolve any domain (no 'domain' field and "
              "skill names carry no prefix); guard degrades to arm-level only")
    probe_skills = {str(row["primary_skill"]) for row in probe_rows}
    if probe_skills - set(by_skill):
        raise ValueError(
            f"Probe skills absent from training: {sorted(probe_skills-set(by_skill))}"
        )
    uncovered_skills = set(by_skill) - probe_skills
    if uncovered_skills:
        print(f"[lpf] WARNING: no probe coverage, neutral utility for: {sorted(uncovered_skills)}")

    # Original domain token fractions (guard reference distribution).
    precomputed = [_row_full_tokens(row) for row in train_rows]
    domain_token_counts: Counter[str] = Counter()
    if all(t is not None for t in precomputed):
        for row, tokens in zip(train_rows, precomputed):
            domain_token_counts[row_domain(row)] += int(tokens)
        domain_fraction_source = "precomputed_fields"
    else:
        print("[v2] tokenizing training rows once for domain reference "
              "fractions (fields prompt_tokens/answer_tokens absent)")
        domain_fraction_source = "startup_tokenization"
    total_ref_tokens = sum(domain_token_counts.values())

    model, tokenizer, device, dtype = load_model_and_tokenizer(
        config["model_name_or_path"],
        revision=str(config.get("model_revision", "main")),
        train_lora=True,
        lora_rank=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        lora_target=config["lora_target"],
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
    )
    if domain_fraction_source == "startup_tokenization":
        for row in train_rows:
            item = tokenize_supervised(row, tokenizer, int(config["max_length"]))
            domain_token_counts[row_domain(row)] += int(
                item["total_tokens"]
            )
        total_ref_tokens = sum(domain_token_counts.values())
    orig_domain_fraction = {
        domain: count / max(1, total_ref_tokens)
        for domain, count in sorted(domain_token_counts.items())
    }
    print(f"[v2] domain reference fractions ({domain_fraction_source}): "
          f"{orig_domain_fraction}")

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    max_steps = int(config["max_optimizer_steps"])
    token_budget = int(config["token_budget"])
    warmup_tokens = int(float(config["warmup_ratio"]) * token_budget)
    scheduler = TokenProgressCosine(
        optimizer,
        base_lr=float(config["learning_rate"]),
        warmup_tokens=warmup_tokens,
        total_tokens=token_budget,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    guard = CoverageGuard(
        arm_domain=arm_domain,
        orig_domain_fraction=orig_domain_fraction,
        window_tokens=int(config.get("coverage_window_tokens", 250000)),
        min_arm_exposure_fraction=float(
            config.get("min_arm_exposure_fraction", 0.005)
        ),
        domain_deviation_tolerance=float(
            config.get("domain_deviation_tolerance", 0.10)
        ),
        enabled=bool(config.get("coverage_guard_enabled", False)),
    )
    hysteresis_enabled = bool(config.get("replay_hysteresis_enabled", False))
    hysteresis = (
        ReplayHysteresis(
            on_threshold=float(config.get("replay_on_threshold", 0.05)),
            off_threshold=float(config.get("replay_off_threshold", 0.02)),
            cooldown_rounds=int(config.get("replay_cooldown_rounds", 2)),
            max_replay_fraction=float(config.get("max_replay_fraction", 0.20)),
        )
        if hysteresis_enabled
        else None
    )

    selector_path = output_dir / "lpf_selector_log.jsonl"
    train_path = output_dir / "train_log.jsonl"
    selector_handle = selector_path.open("w", encoding="utf-8")
    train_handle = train_path.open("w", encoding="utf-8")
    skill_state: dict[str, dict[str, float]] = {}
    last_probe_scores: dict[str, dict[str, float | int]] = {}
    last_selected_round: dict[str, int] = {}
    selection_counts: Counter[str] = Counter()
    exposure_tokens: Counter[str] = Counter()
    reference_sketch: torch.Tensor | None = None
    cached_conflicts: dict[str, float] = {}
    cached_candidate_arms: list[str] = []
    seen_full_tokens = 0
    seen_answer_tokens = 0
    replay_tokens_total = 0
    optimizer_step = 0
    round_index = 0
    partial_final_update = False
    budget_stop_reason: str | None = None
    update_token_values: list[int] = []
    update_partial_count = 0
    timing = Counter()
    wall_start = time.perf_counter()
    try:
        while optimizer_step < max_steps and seen_full_tokens < token_budget:
            round_start = time.perf_counter()
            probe_updated = (
                not skill_state
                or round_index % int(config["probe_interval_rounds"]) == 0
            )
            probe_start = time.perf_counter()
            if probe_updated:
                last_probe_scores = probe_scores(
                    model,
                    tokenizer,
                    probe_rows,
                    int(config["max_length"]),
                    int(config["probe_batch_size"]),
                    device,
                )
                skill_state = update_skill_state(
                    skill_state,
                    last_probe_scores,
                    float(config["ema_fast_decay"]),
                    float(config["ema_slow_decay"]),
                    round_index,
                )
            probe_elapsed = time.perf_counter() - probe_start
            timing["probe_seconds"] += probe_elapsed

            utility_start = time.perf_counter()
            utilities = base_arm_utilities(
                arms,
                skill_state,
                last_selected_round,
                round_index,
                config,
                progress_fraction=seen_full_tokens / max(1, token_budget),
            )
            utility_elapsed = time.perf_counter() - utility_start
            timing["utility_seconds"] += utility_elapsed

            gradient_elapsed = 0.0
            gradient_stats = {}
            candidate_arms = list(utilities)
            current_sketches: dict[str, torch.Tensor] = {}
            gradient_updated = bool(config["gradient_conflict_enabled"]) and (
                round_index % int(config["gradient_interval_rounds"]) == 0
            )
            if gradient_updated:
                gradient_start = time.perf_counter()
                (
                    utilities,
                    current_sketches,
                    candidate_arms,
                    gradient_stats,
                ) = apply_gradient_conflict(
                    model,
                    tokenizer,
                    arms,
                    utilities,
                    reference_sketch,
                    config,
                    device,
                    seed,
                    round_index,
                )
                cached_conflicts = {
                    str(arms[arm][0]["primary_skill"]): utilities[arm][
                        "gradient_conflict"
                    ]
                    for arm in candidate_arms
                }
                cached_candidate_arms = list(candidate_arms)
                gradient_elapsed = time.perf_counter() - gradient_start
                timing["gradient_seconds"] += gradient_elapsed
            elif bool(config["gradient_conflict_enabled"]):
                if cached_candidate_arms:
                    candidate_arms = list(cached_candidate_arms)
                for arm, detail in utilities.items():
                    skill = str(arms[arm][0]["primary_skill"])
                    conflict = cached_conflicts.get(skill, 0.0)
                    detail["gradient_conflict"] = conflict
                    detail["final_utility"] -= (
                        float(config["gradient_conflict_weight"]) * conflict
                    )

            # --- v2 coverage guard: restricts the candidate arm set only ---
            coverage_allowed, coverage_state = guard.allowed_arms(
                sorted(arms), seen_full_tokens
            )

            explored = rng.random() < float(config["epsilon_exploration"])
            if str(config["selector_mode"]) == "random" or explored:
                pool_sel = sorted(set(arms) & set(coverage_allowed))
                if not pool_sel:
                    pool_sel = list(coverage_allowed)
                selected_arm = rng.choice(pool_sel)
            else:
                allowed = candidate_arms if gradient_updated else list(utilities)
                allowed = [a for a in allowed if a in set(coverage_allowed)]
                if not allowed:
                    allowed = list(coverage_allowed)
                selected_arm = max(
                    allowed,
                    key=lambda arm: utilities[arm]["final_utility"],
                )
            selected_skill = str(arms[selected_arm][0]["primary_skill"])
            selection_counts[selected_arm] += 1
            last_selected_round[selected_arm] = round_index

            if selected_skill in current_sketches:
                selected_sketch = current_sketches[selected_skill]
                if reference_sketch is None:
                    reference_sketch = selected_sketch
                else:
                    decay = float(config["gradient_reference_ema"])
                    reference_sketch = (
                        decay * reference_sketch
                        + (1.0 - decay) * selected_sketch
                    )
                    norm = torch.linalg.vector_norm(reference_sketch)
                    if float(norm.item()) > 0:
                        reference_sketch = reference_sketch / norm

            # --- replay decision: hysteresis (v2 candidate) or legacy -------
            replay_skill = max(
                skill_state,
                key=lambda skill: skill_state[skill]["forgetting"],
            )
            max_forgetting = float(skill_state[replay_skill]["forgetting"])
            current_replay_fraction = replay_tokens_total / max(
                1, seen_full_tokens
            )
            if hysteresis is not None:
                replay_active, replay_info = hysteresis.update(
                    max_forgetting, round_index, current_replay_fraction
                )
                replay_active = replay_active and bool(config["replay_enabled"])
            else:
                replay_active = bool(config["replay_enabled"]) and (
                    max_forgetting
                    >= float(config["replay_forgetting_threshold"])
                )
                replay_info = {
                    "state": "legacy_single_threshold",
                    "forgetting": max_forgetting,
                    "threshold": float(
                        config.get("replay_forgetting_threshold", 0.0)
                    ),
                    "capped": False,
                    "transition": "n/a",
                }
            replay_info["replay_fraction_so_far"] = current_replay_fraction

            selector_handle.write(
                json.dumps(
                    {
                        "round": round_index,
                        "optimizer_step_before": optimizer_step,
                        "cumulative_full_tokens_before": seen_full_tokens,
                        "probe_updated": probe_updated,
                        "probe_scores": last_probe_scores,
                        "skill_state": skill_state,
                        "selector_mode": config["selector_mode"],
                        "explored": explored,
                        "selected_arm": selected_arm,
                        "selected_skill": selected_skill,
                        "replay_active": replay_active,
                        "replay_skill": replay_skill,
                        "replay_hysteresis": replay_info,
                        "coverage_state": coverage_state,
                        "arm_utilities": utilities,
                        "gradient_updated": gradient_updated,
                        "gradient_candidate_arms": candidate_arms,
                        "gradient_stats": gradient_stats,
                        "timing_seconds": {
                            "probe": probe_elapsed,
                            "utility": utility_elapsed,
                            "gradient": gradient_elapsed,
                            "round_before_training": (
                                time.perf_counter() - round_start
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            selector_handle.flush()

            train_start = time.perf_counter()
            for _ in range(int(config["steps_per_round"])):
                if optimizer_step >= max_steps or seen_full_tokens >= token_budget:
                    break
                use_replay = replay_active and (
                    rng.random() < float(config["replay_ratio"])
                )
                if use_replay:
                    pool = by_skill[replay_skill]
                    training_source = f"replay::{replay_skill}"
                else:
                    pool = arms[selected_arm]
                    training_source = selected_arm
                stats = train_optimizer_step_v2(
                    model,
                    tokenizer,
                    pool,
                    rng,
                    optimizer,
                    scheduler,
                    device,
                    dtype,
                    int(config["max_length"]),
                    int(config["micro_batch_size"]),
                    float(config["max_grad_norm"]),
                    update_token_target,
                    update_token_tolerance,
                    global_remaining_tokens=token_budget - seen_full_tokens,
                    token_normalized_gradient=token_normalized_gradient,
                    seen_full_tokens=seen_full_tokens,
                )
                if int(stats["full_tokens"]) <= 0:
                    budget_stop_reason = "next_batch_would_overshoot"
                    break
                optimizer_step += 1
                batch_tokens = int(stats["full_tokens"])
                seen_before = seen_full_tokens
                seen_full_tokens += batch_tokens
                seen_answer_tokens += int(stats["answer_tokens"])
                if use_replay:
                    replay_tokens_total += batch_tokens
                partial_final_update = partial_final_update or bool(
                    stats.get("partial_final_update", 0)
                )
                update_token_values.append(batch_tokens)
                update_partial_count += int(stats.get("partial_final_update", 0))
                exposure_tokens[training_source] += batch_tokens
                guard.observe(
                    arm=(None if use_replay else selected_arm),
                    domain=skill_domain.get(
                        replay_skill if use_replay else selected_skill,
                        "unknown",
                    ),
                    tokens=batch_tokens,
                    seen_tokens_before=seen_before,
                )
                train_handle.write(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_step,
                            "round": round_index,
                            "selected_arm": selected_arm,
                            "training_source": training_source,
                            "is_replay": use_replay,
                            "cumulative_full_tokens": seen_full_tokens,
                            "cumulative_answer_tokens": seen_answer_tokens,
                            **stats,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                train_handle.flush()
            train_elapsed = time.perf_counter() - train_start
            timing["train_seconds"] += train_elapsed
            save_interval = int(config.get("save_every_rounds", 0))
            if save_interval > 0 and (round_index + 1) % save_interval == 0:
                checkpoint_start = time.perf_counter()
                model.save_pretrained(
                    output_dir / f"checkpoint-round-{round_index + 1:04d}"
                )
                timing["checkpoint_seconds"] += (
                    time.perf_counter() - checkpoint_start
                )
            timing["round_seconds"] += time.perf_counter() - round_start
            round_index += 1
            if budget_stop_reason is not None:
                break
    finally:
        selector_handle.close()
        train_handle.close()

    wall_seconds = time.perf_counter() - wall_start
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    final_probe_start = time.perf_counter()
    final_scores = probe_scores(
        model,
        tokenizer,
        probe_rows,
        int(config["max_length"]),
        int(config["probe_batch_size"]),
        device,
    )
    final_probe_seconds = time.perf_counter() - final_probe_start
    selector_seconds = (
        timing["probe_seconds"]
        + timing["utility_seconds"]
        + timing["gradient_seconds"]
    )
    update_array = np.array(update_token_values, dtype=np.float64)
    mode = str(config["selector_mode"])
    utility_forgetting_weight = (
        0.0
        if mode in ("no_forgetting", "lp_only", "loss_only", "random")
        else float(config.get("forgetting_weight", 0.0))
    )
    # advisor convention: candidate_dead is auto-recorded in the manifest when
    # the config asks for it and the guard failed to hold domain deviation.
    domain_tol = float(config.get("domain_deviation_tolerance", 0.10))
    final_domain_deviation = {
        d: guard.domain_tokens.get(d, 0) / max(1, guard.total_tokens) - frac
        for d, frac in orig_domain_fraction.items()
    }
    max_final_dev = max(
        (abs(v) for v in final_domain_deviation.values()), default=0.0
    )
    guard_failed = (
        bool(config.get("coverage_guard_enabled", False))
        and max_final_dev > domain_tol
    )
    candidate_dead = (
        bool(config.get("candidate_dead_if_guard_fails", False)) and guard_failed
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "LPF-Curriculum",
        "protocol_family": REQUIRED_PROTOCOL,
        "hyperparameter_provenance": config.get(
            "hyperparameter_provenance", "arbitrary_but_frozen"
        ),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "train_sha256": sha256_file(config["train_file"]),
        "selector_probe_sha256": sha256_file(
            config["selector_probe_file"]
        ),
        "leakage_check_train_x_probe": len(train_ids & probe_ids),
        "arms_without_probe_coverage": sorted(uncovered_skills),
        "model_revision_loaded": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "dtype": str(dtype),
        "optimizer_steps": optimizer_step,
        "rounds": round_index,
        "termination_reason": (
            "token_budget_exact"
            if seen_full_tokens >= token_budget or budget_stop_reason is not None
            else "max_optimizer_steps"
        ),
        "trained_full_tokens": seen_full_tokens,
        "trained_answer_tokens": seen_answer_tokens,
        "target_full_tokens": token_budget,
        "budget_utilization": seen_full_tokens / max(token_budget, 1),
        "overshoot_tokens": max(0, seen_full_tokens - token_budget),
        "partial_final_update": partial_final_update,
        "budget_stop_reason": budget_stop_reason,
        "lr_schedule": "token_progress_cosine",
        "lr_base": float(config["learning_rate"]),
        "lr_warmup_tokens": warmup_tokens,
        "lr_total_tokens": token_budget,
        "update_token_target": update_token_target,
        "update_token_tolerance": update_token_tolerance,
        "token_normalized_gradient": token_normalized_gradient,
        "update_token_stats": {
            "n_updates": int(len(update_token_values)),
            "mean": float(update_array.mean()) if len(update_array) else 0.0,
            "std": float(update_array.std()) if len(update_array) else 0.0,
            "min": int(update_array.min()) if len(update_array) else 0,
            "max": int(update_array.max()) if len(update_array) else 0,
            "n_partial": int(update_partial_count),
        },
        "utility_forgetting_weight": utility_forgetting_weight,
        "replay_forgetting_enabled": bool(config.get("replay_enabled", False)),
        "replay_hysteresis_enabled": hysteresis_enabled,
        "replay_hysteresis_params": (
            {
                "on_threshold": hysteresis.on,
                "off_threshold": hysteresis.off,
                "cooldown_rounds": hysteresis.cooldown_rounds,
                "max_replay_fraction": hysteresis.max_fraction,
            }
            if hysteresis is not None
            else None
        ),
        "replay_tokens_total": replay_tokens_total,
        "replay_fraction": replay_tokens_total / max(1, seen_full_tokens),
        "replay_hysteresis_transitions": (
            hysteresis.transitions if hysteresis is not None else []
        ),
        "coverage_guard_enabled": bool(config.get("coverage_guard_enabled", False)),
        "coverage_guard_params": {
            "window_tokens": int(config.get("coverage_window_tokens", 250000)),
            "min_arm_exposure_fraction": float(
                config.get("min_arm_exposure_fraction", 0.005)
            ),
            "domain_deviation_tolerance": float(
                config.get("domain_deviation_tolerance", 0.10)
            ),
        },
        "orig_domain_fraction": orig_domain_fraction,
        "orig_domain_fraction_source": domain_fraction_source,
        "domain_exposure_tokens": dict(guard.domain_tokens),
        "final_domain_deviation": final_domain_deviation,
        "candidate_dead": candidate_dead,
        "selection_counts_by_arm": dict(selection_counts),
        "exposure_tokens_by_source": dict(exposure_tokens),
        "final_selector_probe_scores": final_scores,
        "timing_seconds": {
            **dict(timing),
            "selector_total": selector_seconds,
            "training_wall": wall_seconds,
            "final_probe_excluded_from_overhead": final_probe_seconds,
        },
        "selector_overhead_ratio": selector_seconds
        / max(wall_seconds, 1e-8),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "selector_log_sha256": sha256_file(selector_path),
        "train_log_sha256": sha256_file(train_path),
        "interpretation_warning": (
            "Positive forgetting raises replay priority. Coverage guard and "
            "replay hysteresis restrict the candidate set / replay timing only; "
            "utility weights are never modified online."
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
