#!/usr/bin/env python3
"""gen_week8_configs.py — Day 50: Week-8 configs + machine-readable prereg.

Single source of truth = YOUR two yaml files in campus_week8_9_starter/configs/:
  - week8_lp_match_replay.yaml      (candidate spec, advisor version)
  - preregistration_week8_9.yaml    (frozen gates / seeds / stop rules)

What this script does:
  1. Loads the candidate spec and the prereg skeleton.
  2. Loads the VERIFIED parent configs from Week6/7 run manifests
     (campus_week6_7_starter/saves/...), never hand-written values.
  3. Asserts the candidate spec matches the a2 parent on every
     inheritance-listed field (README8_9 §2.3); any mismatch = HARD FAIL
     with a diff table (nothing is generated silently).
  4. Adds replay_hysteresis_enabled: true (the spec has the four hysteresis
     knobs but no enable flag — without it the trainer falls back to the
     legacy single threshold).
  5. Writes 12 full configs (4 methods x seeds 22/23/24) + 4 smoke configs
     into configs/, and reports/week8/preregistration_week8.json whose
     frozen_gates are built FROM the prereg yaml.
  6. Prints unchanged_fields_sha256 for you to paste into the yaml's
     `fill_before_run` slot.

Run:
    cd /root/autodl-tmp/campus_week8_9_starter && python3 scripts/gen_week8_configs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

NEW_ROOT = Path("/root/autodl-tmp/campus_week8_9_starter")
OLD_ROOT = Path("/root/autodl-tmp/campus_week6_7_starter")
SMOKE_BUDGET = 200_000
BOOTSTRAP_RNG = 20250814  # not present in the prereg yaml; fixed here, printed

# README8_9 §2.3 inheritance list: candidate must equal a2 on these fields
INHERIT_KEYS = [
    "model_name_or_path", "model_revision", "train_file", "selector_probe_file",
    "lora_rank", "lora_alpha", "lora_dropout", "lora_target",
    "learning_rate", "warmup_ratio", "weight_decay", "max_grad_norm",
    "max_length", "micro_batch_size", "gradient_accumulation_steps",
    "steps_per_round", "probe_batch_size", "probe_interval_rounds",
    "ema_fast_decay", "ema_slow_decay", "competence_margin",
    "epsilon_exploration", "replay_enabled", "replay_ratio",
    "replay_forgetting_threshold", "token_budget",
]
# protocol keys come from the spec yaml and apply to ALL four methods
PROTOCOL_KEYS = [
    "protocol_family", "strict_token_budget", "update_token_target",
    "update_token_tolerance", "token_normalized_gradient",
    "token_progress_cosine", "max_optimizer_steps",
]
SOFT_KEYS = ["save_every_rounds"]  # checkpoint cadence: reported, not asserted

METHODS = {
    "random_sameloop": {
        "parent_glob": "saves/week6/*random*sameloop*seed20250719*",
        "extra": {},
    },
    "lp_only_replay": {
        "parent_glob": "saves/week7/a2_lp_only_seed20250719*",
        "extra": {},
    },
    "lp_match_replay": None,  # built from the candidate spec, not a manifest
    "lpf_sparse": {
        "parent_glob": "saves/week6/*lpf*sparse*seed20250719*",
        "extra": {},
    },
}

CONTROL_OVERRIDES = {
    "replay_hysteresis_enabled": False,
    "coverage_guard_enabled": False,
}


def find_parent(pattern: str) -> Path:
    matches = sorted(OLD_ROOT.glob(pattern))
    matches = [m for m in matches if (m / "run_manifest.json").exists()]
    if len(matches) != 1:
        print(f"[FATAL] parent glob {pattern!r} matched {len(matches)} dirs")
        print("        available:")
        for p in sorted(OLD_ROOT.glob("saves/*/*")):
            print(f"        - {p.relative_to(OLD_ROOT)}")
        sys.exit(1)
    return matches[0]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=NEW_ROOT)
    args = parser.parse_args()
    root: Path = args.root
    configs_dir = root / "configs"
    reports_dir = root / "reports" / "week8"
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    spec_path = configs_dir / "week8_lp_match_replay.yaml"
    prereg_yaml_path = configs_dir / "preregistration_week8_9.yaml"
    for path in (spec_path, prereg_yaml_path):
        if not path.exists():
            print(f"[FATAL] 缺文件: {path}")
            print("        先把你的两个 yaml 传到 campus_week8_9_starter/configs/")
            return 1
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    prereg_yaml = yaml.safe_load(prereg_yaml_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in prereg_yaml["new_seeds"]]

    # ---- parents from verified manifests ------------------------------------
    parents: dict[str, dict] = {}
    parent_info: dict[str, dict] = {}
    for method, mspec in METHODS.items():
        if mspec is None:
            continue
        run_dir = find_parent(mspec["parent_glob"])
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        parents[method] = manifest["config"]
        parent_info[method] = {
            "source_manifest": str(run_dir / "run_manifest.json"),
            "config_sha256": manifest.get("config_sha256"),
            "train_sha256": manifest.get("train_sha256"),
            "selector_probe_sha256": manifest.get("selector_probe_sha256"),
        }
        print(f"[parent] {method:16s} <- {run_dir.relative_to(OLD_ROOT)}")
    a2 = parents["lp_only_replay"]

    # ---- inheritance assertion: candidate spec vs a2 parent ------------------
    hard_fails: list[str] = []
    notes: list[str] = []
    diff_rows = []
    for key in INHERIT_KEYS:
        spec_v = spec.get(key, "<absent>")
        parent_v = a2.get(key, "<absent>")
        same = spec_v == parent_v
        diff_rows.append((key, parent_v, spec_v, same))
        if not same:
            hard_fails.append(f"{key}: a2={parent_v!r} vs spec={spec_v!r}")
    for key in SOFT_KEYS:
        if spec.get(key) != a2.get(key):
            notes.append(
                f"{key}: a2={a2.get(key)!r} vs spec={spec.get(key)!r} "
                "(仅影响 checkpoint 节奏, 不影响优化数学, 用 spec 值)"
            )
    print("\n== 继承断言 (候选 spec vs a2 父配置) ==")
    print(f"{'key':32s} {'a2 实际值':>24s} {'spec 值':>24s}  判定")
    for key, parent_v, spec_v, same in diff_rows:
        print(f"{key:32s} {str(parent_v):>24s} {str(spec_v):>24s}  "
              f"{'✓' if same else '✗ 不一致'}")
    if "replay_hysteresis_enabled" not in spec:
        notes.append(
            "spec 缺 replay_hysteresis_enabled — 已自动补 true "
            "(否则滞回不生效, 落回单阈值 replay_forgetting_threshold="
            f"{spec.get('replay_forgetting_threshold')})"
        )
    # probe_interval_rounds 不做硬编码数值检查: 继承断言 (spec == a2 实际值)
    # 已覆盖它。md §2.3 曾误记 a2=4, manifest 实际=1 —— 以对齐 a2 实际值为准。
    if hard_fails:
        print("\n[HARD FAIL] 候选 spec 与 a2 父配置在继承字段上不一致:")
        for f in hard_fails:
            print(f"  - {f}")
        print("未生成任何配置。两种合法处理: (a) 把 spec 改成 a2 的实际值; "
              "(b) 若 md 继承表本身要改, 先改 README 再回来。把上表贴回。")
        return 1

    # ---- build configs --------------------------------------------------------
    protocol_block = {k: spec[k] for k in PROTOCOL_KEYS if k in spec}
    protocol_block["hyperparameter_provenance"] = spec.get(
        "hyperparameter_source", "arbitrary_but_frozen"
    )
    if int(protocol_block.get("max_optimizer_steps", 0)) < 950:
        notes.append(
            f"max_optimizer_steps={protocol_block.get('max_optimizer_steps')}: "
            "1.5M/1638 ≈ 917 次 update 上界, 非绑定 ✓"
        )

    def write_config(name: str, cfg: dict) -> str:
        text = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)
        (configs_dir / name).write_text(text, encoding="utf-8")
        return sha256_text(text)

    entries = []
    for method, mspec in METHODS.items():
        for seed in seeds:
            if method == "lp_match_replay":
                cfg = dict(a2)          # parent first (§2.3)
                cfg.update(spec)        # spec overlays (inheritance asserted ==)
                cfg["replay_hysteresis_enabled"] = True
            else:
                cfg = dict(parents[method])
                cfg.update(CONTROL_OVERRIDES)
                cfg.update(mspec["extra"])
            cfg.update(protocol_block)
            cfg["gradient_conflict_enabled"] = False  # §4.4: conflict removed
            cfg["seed"] = seed
            cfg["output_dir"] = f"saves/week8/{method}_seed{seed}"
            name = f"week8_{method}_seed{seed}.yaml"
            sha = write_config(name, cfg)
            entries.append(
                {
                    "run_id": f"{method}_seed{seed}",
                    "phase": "screen" if seed == seeds[0] else "confirm",
                    "method": method,
                    "seed": seed,
                    "config": f"configs/{name}",
                    "config_sha256": sha,
                    "output_dir": cfg["output_dir"],
                    "cmd": f"python3 scripts/train_lpf_v2.py --config configs/{name}",
                }
            )
        # smoke variant (first seed, 200k budget)
        if method == "lp_match_replay":
            cfg = dict(a2)
            cfg.update(spec)
            cfg["replay_hysteresis_enabled"] = True
        else:
            cfg = dict(parents[method])
            cfg.update(CONTROL_OVERRIDES)
            cfg.update(mspec["extra"])
        cfg.update(protocol_block)
        cfg["gradient_conflict_enabled"] = False
        cfg["seed"] = seeds[0]
        cfg["token_budget"] = SMOKE_BUDGET
        cfg["output_dir"] = f"saves/week8_smoke/{method}"
        name = f"week8_smoke_{method}.yaml"
        sha = write_config(name, cfg)
        entries.append(
            {
                "run_id": f"smoke_{method}",
                "phase": "smoke",
                "method": method,
                "seed": seeds[0],
                "config": f"configs/{name}",
                "config_sha256": sha,
                "output_dir": cfg["output_dir"],
                "cmd": f"python3 scripts/train_lpf_v2.py --config configs/{name}",
            }
        )

    # ---- frozen gates from the prereg yaml (single source of truth) ----------
    pb = prereg_yaml.get("paired_bootstrap", {})
    sr = prereg_yaml.get("stop_rules", {})
    frozen_gates = {
        "noninferiority": {
            "primary_comparison": prereg_yaml.get(
                "primary_comparison", "LP-Match-Replay - Random-SameLoop"
            ),
            "secondary_comparison": prereg_yaml.get(
                "secondary_comparison", "LP-Match-Replay - LPF-Sparse"
            ),
            "metric": "gsm8k_strict",
            "delta_point_min_pp": float(
                pb.get("noninferiority_point_threshold_pp", -1.0)
            ),
            "ci_lower_min_pp": float(
                pb.get("noninferiority_ci_lower_threshold_pp", -2.5)
            ),
            "bootstrap_B": int(pb.get("resamples", 10000)),
            "bootstrap_rng": BOOTSTRAP_RNG,
            "bootstrap_unit": "paired_by_doc_id_within_seed_then_averaged",
        },
        "base_rule": {
            "all_seeds_above_base": True,
            "no_seed_below_base_minus_pp": abs(
                float(sr.get("any_seed_below_base_by_pp", -2.0))
            ),
        },
        "stability": {
            "max_arm_share_lt": float(sr.get("max_arm_share", 0.50)),
            "domain_deviation_le": float(sr.get("domain_deviation", 0.10)),
            "replay_fraction_le": float(sr.get("replay_fraction", 0.20)),
            "overhead_le": sr.get("selector_overhead_ratio", 0.10),
        },
        "candidate_death_rules": [
            "不改冻结数值(match/staleness/replay/coverage/epsilon 全部旋钮)",
            "不换 seed",
            "不按测试集反调数据",
            "候选失败即 candidate_dead=true, 转入路线 B, 保留完整失败记录",
        ],
        "smoke_failure_fallback": sr.get(
            "smoke_failure_fallback", "protocol_regression_random_sparse_only"
        ),
    }

    unchanged = {k: a2.get(k) for k in INHERIT_KEYS}
    unchanged_sha = hashlib.sha256(
        json.dumps(unchanged, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    train_shas = {p["train_sha256"] for p in parent_info.values()}
    probe_shas = {p["selector_probe_sha256"] for p in parent_info.values()}
    if len(train_shas) != 1 or len(probe_shas) != 1:
        print(f"[FATAL] 父配置数据/probe SHA 不一致: {train_shas} {probe_shas}")
        return 1

    prereg = {
        "phase": "week8",
        "protocol_family": "token_update_matched_v2",
        "source_yaml": "configs/preregistration_week8_9.yaml",
        "seeds": seeds,
        "smoke_seed": seeds[0],
        "smoke_budget": SMOKE_BUDGET,
        "frozen_gates": frozen_gates,
        "frozen_hyperparameters": prereg_yaml.get("frozen_hyperparameters", {}),
        "worst_skill_definition": prereg_yaml.get("worst_skill_definition", {}),
        "hyperparameter_provenance": {
            "mode": prereg_yaml.get("frozen_hyperparameters", {}).get(
                "source", "arbitrary_but_frozen"
            ),
            "note": (
                "若改用 derived_from_week67, 必须在任何 v2 run 启动前完成 "
                "reports/week8/hyperparameter_derivation.md; 看完分数后不得再选阈值。"
            ),
        },
        "unchanged_fields_sha256": unchanged_sha,
        "parent_configs": parent_info,
        "train_sha256": sorted(train_shas)[0],
        "selector_probe_sha256": sorted(probe_shas)[0],
        "entries": entries,
    }
    out = reports_dir / "preregistration_week8.json"
    out.write_text(json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8")

    for note in notes:
        print(f"[note] {note}")
    print("\n== 生成结果 ==")
    print(f"configs: {len(seeds) * len(METHODS)} 正式 + {len(METHODS)} 烟雾 "
          f"→ {configs_dir}")
    print(f"prereg:  {out} ({len(entries)} entries)")
    print(f"\nunchanged_fields_sha256 = {unchanged_sha}")
    print("  → 请把这个值填进 configs/preregistration_week8_9.yaml 的 "
          "candidate_inheritance.unchanged_fields_sha256")
    print("\n首条命令预览:")
    print(" ", entries[0]["cmd"])
    print("\n[OK] Day50 完成。把继承断言表 + 本页贴回确认后再进 Day51。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
