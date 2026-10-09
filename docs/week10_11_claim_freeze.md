# Week10–11 Claim Freeze

- Title: When Competence-Aware Curriculum Fails: strict token/update auditing of CAMPUS scheduling and low-cost diagnosis
- Frozen at UTC: 2026-09-21T19:09:24.554579+00:00
- Route: B
- Protocol rule: v1/v2/transfer runs are never mixed in one effect estimate.
- Statistical rule: seed-level summaries are descriptive; confirmatory CI must use the preregistered paired bootstrap artefact.

## Audit status

- Run artifact audits: 3 pass, 0 fail.
- Sample audits with warnings/failures: 3.

## Claims allowed in the main text

1. Under strict token/update matching, Random-SameLoop is a strong baseline in the studied 0.5B LoRA setting.
2. The frozen LP-Match-Replay candidate is not non-inferior to Random under its preregistered gate.
3. Sparse probing reduces selector cost and is an engineering/diagnostic result, not proof of an average-score gain.
4. Gradient conflict is retained as a mechanism analysis and is not claimed as a useful scheduling signal.

## Claims prohibited without new preregistration

- competence-aware scheduling is universally ineffective.
- backward curriculum is universally superior.
- probe loss is a validated proxy for benchmark ability.
- any positive single-seed result is a confirmed improvement.

## Inputs

- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250725_artifact_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250726_artifact_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250727_artifact_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250725_gsm8k_samples_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250726_gsm8k_samples_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_seed20250727_gsm8k_samples_audit.json
- /root/autodl-tmp/campus_project/reports/week11/safe_fallback_aggregate.json

## Aggregate snapshot

~~~json
{
  "LPF-Curriculum": {
    "n": 3,
    "mean": 0.2423553196866313,
    "sd": 0.007891076571137724,
    "seed_block_bootstrap_95ci": [
      0.23351023502653526,
      0.2486732373009856
    ]
  }
}
~~~

---

## Addendum A（Week 12 勘误附录，2026-09-22）

Week 11 冻结报告中"Safe 全程零回退（0/245）"的表述为审计链读错字段所致的测量错误，现更正：

- 事件级重计数：22/245 轮发生回退（9.0%；seed25 11/82、seed26 6/82、seed27 5/81；smoke 3/12）
- 回退轮与零激活轮完全一致（fallback==zero_active 于全部 run），符合"候选集为空 → Random 回退"设计语义
- Gate 确实执行了过滤：每轮激活技能中位数 5/17（范围 0–16）
- 选择越出激活集仅 9 轮，全部为 replay_active+explored 旁路（不构成对 SafeGate 的证伪）
- Week 11 基于"零回退"的推断（如"gate 未起作用"）一并作废；正文主张以 Week 12 修订表为准：week12_results_20260922_033907/week12_claim_revision.md
- 证据：gate_checks/gate_compact_summary.json、gate_verdict.md（生成脚本 sha256 见 server_schema_notes.md）

本附录为追加勘误，原 Week 11 正文不作改动。
