# Week 12 放行清单（对照 README12 §8.1）

- [x] 15 个 run 身份、协议、seed、配置来源可追溯 → evidence_index.json（15/15，含 manifest sha256、config provenance、samples 路径）
- [x] Gate 首轮差异已解释 → gate_verdict.md 第 5 条（首轮 1 次观测 < confirm_rounds=2，预期行为，合成单测复现）
- [x] fallback 缺失值未被当作 False → inspect 输出 all_events_have_boolean_fallback=true；state_missing 为零（gate_checks/*_gate.json）
- [x] "零fallback=零过滤"错误推理已删除 → gate_verdict.md：22/245 回退 + 激活中位 5/17 + 9 个 replay/exploration 旁路轮分列
- [x] GSM8K 官方分数复算一致，filter 重复未入样本数 → normalized_tests.json t1/t3（15/15，容差 5e-5，N=1319/run）
- [x] 原始文本格式诊断与官方 strict/flexible 分开 → quadrant_counts.json + week12_claim_revision.md（"答案后续写"只影响 flexible；旧 format_failure_rate 标 deprecated）
- [x] CI 采样单位、seed 限制与 reference 条件性说明清楚 → week12_claim_revision.md 统计口径披露（含 ~0.05pp MC 端点差异记录）
- [x] IFEval、技能与成本指标已核实/降级 → week12_main_table.md v2（IFEval prompt/inst 分列）；技能指标降级：worst-skill/forgetting=N/A（最终独立 analysis probe 未跑，selector probe 分数按 §6.2 不冒充；forgetting_area=N/A 不拿首末两点拟合）
- [x] 新分析、历史登记、重建产物与勘误分开标识 → protocol_deviations.md 7 条 + week12_claim_revision.md 修订表
- [x] 表格可由冻结输入重新生成 → 生成脚本 sha256 全部登记于 server_schema_notes.md；gate_compact REPRODUCE_CHECK=IDENTICAL

## 最终状态判定：Ready for manuscript
主分数、样本配对、方法身份、机制核验全部通过；上述降级项写入论文限制节。
负结果（route B）不是阻塞条件；图按计划排 Week 13（README12 §9）。

## Day75 补充（reviewer 三组材料请求答复后）
- 偏差 #8（跨 era 主机/软件差异，含 lm-eval 0.4.12 vs 0.4.8 补充）与 #9（fewshot_seed 差异）已入 protocol_deviations.md
- 代码身份、seed 三来源、probe 间隔三时期、评测口径、环境自记录已入 server_schema_notes.md（Day75 blocks）
- 维持限制节：PEFT 版本未留存、Safe era lm-eval 0.4.8（推断级来源）、worst-skill/forgetting（analysis probe 未留存）
