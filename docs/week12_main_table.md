# Week 12 主结果表 v2（输入：week12_main_stats.json + 各 run run_manifest.json / eval_results.json）

协议 token_update_matched_v2；主指标 GSM8K strict-match（官方 primary filter）；overhead = manifest selector_overhead_ratio（分子 probe+utility 计时，分母总训练墙钟）

| method | n | GSM8K strict | flexible | IFEval prompt-strict | IFEval inst-strict | full tok | answer tok | updates | util | overhead | wall s | replay frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random-SameLoop | 3 | 0.2782+-0.0134 | 0.2798+-0.0127 | 0.2539 | 0.3493 | 1499977 | 754276 | 818 | 0.99998 | 0.072 | 11302 | 0.173 |
| LP-only+replay | 3 | 0.2709+-0.0506 | 0.2707+-0.0502 | 0.2384 | 0.3449 | 1499970 | 781445 | 811 | 0.99998 | 0.250 | 13412 | 0.193 |
| LP-Match-Replay | 3 | 0.2418+-0.0266 | 0.2418+-0.0258 | 0.2329 | 0.3265 | 1499975 | 725773 | 807 | 0.99998 | 0.259 | 12856 | 0.185 |
| LPF-Sparse | 3 | 0.2472+-0.0227 | 0.2494+-0.0231 | 0.2779 | 0.3785 | 1499960 | 805363 | 844 | 0.99997 | 0.059 | 14343 | 0.200 |
| Safe (Week11) | 3 | 0.2424+-0.0079 | 0.2143+-0.0374 | 0.2175 | 0.3441 | 1499968 | 718134 | 811 | 0.99998 | 0.309 | 11104 | 0.176 |

## 逐 run strict（附录）
- Random-SameLoop: [0.2638, 0.2805, 0.2904]
- LP-only+replay: [0.3192, 0.2752, 0.2183]
- LP-Match-Replay: [0.2472, 0.213, 0.2654]
- LPF-Sparse: [0.2472, 0.2699, 0.2244]
- Safe (Week11): [0.2449, 0.2487, 0.2335]

## 统计比较（week12_main_stats.json）
- safe_fallback_minus_random_sameloop: -3.589pp CI [-5.231, -1.946] (primary_preregistered)
- lp_match_replay_minus_random_sameloop: -3.639pp CI [-5.155, -2.098] (reproduction_of_registered)
- safe_fallback_minus_lp_match_replay: +0.051pp CI [-1.516, +1.592] (secondary_not_equivalence)
- safe_fallback_minus_lp_only_replay: -2.856pp CI [-4.423, -1.264] (exploratory_vs_actual_parent)
- 敏感性(事后): -3.564pp CI [-6.065, -1.011]

## artifact status / provenance
- 15/15 run: run_manifest.json + eval_results.json + samples + config.yaml 齐备；identity 见 evidence_index.json
- Week 8 runs config_provenance=original；Safe runs=reconstructed_registered_day62
- 缺失/降级项: 无
