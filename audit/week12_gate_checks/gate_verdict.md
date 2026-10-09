# Week 12 SafeGate 核验结论（Day 72）

依据：gate_compact_summary.json（inspect_week12.py 工具 sha 1edc3bdb）、
safe_selector.py sha 583bbbcb…（合成单测 PASS）、safe_reference_random_v2.json sha be63f117…。
工具输出非真实性认证；本文件仅报告日志可支持的最弱充分结论。

1. 零回退：**corrected**。正式 3 run 共 22/245 轮回退（seed25 11/82、seed26 6/82、seed27 5/81），
   smoke 3/12。Week 11 登记的“0/245、门从未触发”为审计链字段误读，作废并勘误。
2. 候选过滤：**observed**（技能级）。门后允许集合 = active_skills，中位 5/17 技能（范围 0–16）；
   资格级 eligible_skills 中位 5–8。回退与门状态完全一致：zero_active 数与回退数逐 run 相等，
   有激活技能时回退为零。
3. 对选择的影响：**observed**。非回退轮次选中臂全部属于激活技能；
   例外 9 轮（seed25×3、seed26×4、seed27×2）均为 replay_active=True 且 explored=True，
   属 replay/exploration 设计旁路（状态字 legacy_single_threshold），非门泄漏。
4. 与父策略等价：**未证明**（按手册通常只能如此写；探索性比较见 statistics，
   safe − lp_only_replay = −2.856pp，方向相反，进一步不支持等价）。
5. smoke/正式首轮差异：**已解释**。首轮 LP 仅 1 次观测，未达 confirm_rounds=2，
   预期回退；服务器实现行为与合成单测一致。
6. 实现语义：技能一次未达标即退出并进入 cooldown（2 轮）。与部分 README
   “连续两次失败才退出”的表述不符 → 判为文档错误，记入 protocol_deviations，
   不以现测试期望重写历史登记。
7. reference 覆盖：17/18 技能。math.algebra 无 probe、无下界、永不达标，
   其 3 个难度臂在全部 4 个 run 中永久不可选（结构性排除 5.6% 臂空间）。
8. 阈值实证：数学 5 技能下界 +0.07~+0.14、达标率 0–17%（seed27 counting_geometry 0.0%）；
   负下界技能达标率 66–86%。“阈值<0 ⇒ 宽松”不成立，实际通过/过滤分布见 per_skill 表。
