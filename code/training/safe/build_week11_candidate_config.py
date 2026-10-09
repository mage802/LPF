import yaml
from pathlib import Path

parent = Path("/root/autodl-tmp/campus_project/reports/week10/week8_lp_only_replay.canonical.yaml")
candidate = Path("/root/autodl-tmp/campus_week10_11_starter/configs/week11_safe_curriculum.yaml")

cfg = yaml.safe_load(parent.read_text(encoding="utf-8")) or {}

# 仅应用已登记字段变化（与 config_inheritance_audit.py 默认白名单一一对应）
registered = {
    "output_dir": "saves/week11/safe_fallback_seed20250725",
    "seed": 20250725,
    "selector_mode": "safe_fallback",
    "safe_candidate_mode": "lp_only",
    "safe_reference_file": "reports/week10/safe_reference_random_v2.json",
    "safe_progress_margin": 0.0,
    "safe_confirm_rounds": 2,
    "safe_fail_cooldown_rounds": 2,
    "run_role": "P1_safe_fallback_confirmation",
    "schema_version": "week11_safe_curriculum_v1",
    "study_title": "When Competence-Aware Curriculum Fails: strict token/update auditing of CAMPUS scheduling and low-cost diagnosis",
}
cfg.update(registered)

candidate.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"written: {candidate}")
