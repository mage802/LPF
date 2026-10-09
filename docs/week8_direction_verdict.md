# Week8 方向裁决 (token_update_matched_v2)

- base GSM8K strict = 11.14 (source: manual_frozen_week6_value)
- seeds = [20250722, 20250723, 20250724]

## 主表 (GSM8K strict %)

| method | s22 | s23 | s24 | mean | sd | 95% CI (t,n=3) |
|---|---|---|---|---|---|---|
| random_sameloop | 26.38 | 28.05 | 29.04 | 27.82 | 1.34 | [24.49, 31.16] |
| lp_only_replay | 31.92 | 27.52 | 21.83 | 27.09 | 5.06 | [14.53, 39.65] |
| lp_match_replay | 24.72 | 21.30 | 26.54 | 24.18 | 2.66 | [17.59, 30.78] |
| lpf_sparse | 24.72 | 26.99 | 22.44 | 24.72 | 2.27 | [19.07, 30.37] |

辅指标 (不作判定): flexible 与 IFEval 见 week9_effect_sizes.json。

## 配对 bootstrap Δ (主比较, 预注册)

- lp_match_replay − random_sameloop: Δ=-3.64pp 95% CI [-5.13, -2.17]  逐种子 {20250722: -1.67, 20250723: -6.75, 20250724: -2.5}
- lp_match_replay − lpf_sparse (次要): Δ=-0.53pp 95% CI [-2.07, +0.96]

## 门禁 (全部来自 preregistration_week8.json frozen_gates)

| 门禁 | 判定 | 实际值 |
|---|---|---|
| 点估计 Δ_mean ≥ -1.0pp | ❌ | Δ=-3.64pp |
| CI 下限 ≥ -2.5pp | ❌ | CI=[-5.13, -2.17] |
| 三个 seed 均高于 Base | ✅ | min=21.30 vs base=11.14 |
| 没有 seed 低于 Base 2pp 以上 | ✅ | min=21.30 |
| max arm share < 0.5 | ✅ | max=0.195 |
| domain deviation ≤ 0.1 | ❌ | max=0.181 |
| replay fraction ≤ 0.2 | ✅ | max=0.1974 |
| overhead <= lp_only control (relative, same protocol) | ❌ | cand=[0.2552, 0.2384, 0.2834] ctrl=[0.2791, 0.2511, 0.2195] |

## 裁决

**路线 B: 非劣效判据未全部满足, candidate_dead=true**

主张模板: 在严格 token 与 update 公平协议下, 随机采样是强且稳定的基线; 复杂能力感知调度可能因代理目标失配和课程塌缩而有害。稀疏 probe 仍提供约 70% 的 selector 成本下降, 项目贡献转为「低成本能力审计 + 课程失败边界」。

纪律: 不改冻结数值 / 不换 seed / 不按测试集反调数据; 失败记录完整保留。
