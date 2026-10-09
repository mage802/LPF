# Week 9 增补：前向/反向静态课程 — 分析 addendum

登记: configs/preregistration_week9_static.yaml (sha256 2c0c74618a02…)
口径: gsm8k strict-match; 配对 bootstrap B=10000 rng=20250814 (与 Week-8 verdict 一致)

## 分数表 (mean ± sd, n=3 种子)

| arm | seed22 | seed23 | seed24 | mean ± sd |
|---|---|---|---|---|
| forward_static | 24.56 | 27.60 | 27.07 | 26.41 ± 1.62 |
| backward_static | 28.66 | 26.76 | 29.49 | 28.30 ± 1.40 |
| random_sameloop | 26.38 | 28.05 | 29.04 | 27.82 ± 1.34 |
| lp_only_replay | 31.92 | 27.52 | 21.83 | 27.09 ± 5.06 |
| lp_match_replay | 24.72 | 21.30 | 26.54 | 24.18 ± 2.66 |
| lpf_sparse | 24.72 | 26.99 | 22.44 | 24.72 ± 2.27 |

## 效应量 (primary)

| comparison | Δ point (pp) | 95% CI | per-seed Δ |
|---|---|---|---|
| forward_static - random_sameloop | -1.42 | [-2.88, +0.05] | 20250722: -1.82, 20250723: -0.45, 20250724: -1.97 |
| backward_static - random_sameloop | +0.48 | [-0.94, +1.90] | 20250722: +2.27, 20250723: -1.29, 20250724: +0.45 |

## 效应量 (secondary)

| comparison | Δ point (pp) | 95% CI | per-seed Δ |
|---|---|---|---|
| forward_static - backward_static | -1.90 | [-3.34, -0.48] | 20250722: -4.09, 20250723: +0.83, 20250724: -2.43 |
| lp_match_replay - forward_static | -2.22 | [-3.77, -0.73] | 20250722: +0.15, 20250723: -6.29, 20250724: -0.53 |
| lp_match_replay - backward_static | -4.12 | [-5.64, -2.65] | 20250722: -3.94, 20250723: -5.46, 20250724: -2.96 |

## 课表保真 (登记的 sanity checks)

| run | clock_dev (<1e-9) | Spearman ρ | n 轮 |
|---|---|---|---|
| forward_static_seed20250722 | 0.0e+00 | +0.936 | 83 |
| forward_static_seed20250723 | 0.0e+00 | +0.936 | 83 |
| forward_static_seed20250724 | 0.0e+00 | +0.934 | 83 |
| backward_static_seed20250722 | 0.0e+00 | -0.936 | 83 |
| backward_static_seed20250723 | 0.0e+00 | -0.936 | 83 |
| backward_static_seed20250724 | 0.0e+00 | -0.937 | 82 |

## 描述性统计 (unguarded drift / replay / 开销)

| run | max域偏差 | replay_frac | 选择器开销 | util |
|---|---|---|---|---|
| forward_static_seed20250722 | 0.198 | 0.1864 | 0.2227 | 0.9999 |
| forward_static_seed20250723 | 0.199 | 0.1941 | 0.2242 | 1.0000 |
| forward_static_seed20250724 | 0.194 | 0.1812 | 0.2224 | 1.0000 |
| backward_static_seed20250722 | 0.168 | 0.1672 | 0.2219 | 1.0000 |
| backward_static_seed20250723 | 0.183 | 0.1880 | 0.2266 | 0.9999 |
| backward_static_seed20250724 | 0.190 | 0.1773 | 0.2247 | 1.0000 |

## 备注
- 本页只报告效应量与 CI；无通过/失败门 (登记 gates: none)。
- Week-8 对照数字出自已冻结的 reports/week8/ 结果。
- 域偏差为 guard 关闭下的自然漂移 (接 Week-8 unguarded-drift 发现)。
