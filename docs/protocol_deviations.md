# Week 12 协议偏差与披露清单

1. **Week 11 回退计数错误（勘误）**：Week 11 审计链报告 fallback 0/82/82/81，
   经 Week 12 事件级重数为 11/82、6/82、5/81。原因为 Week 11 统计读了错误字段/路径。
   主统计结论（route B）不受影响；机制结论已改写。
2. **文档与实现差异**：部分 README 写“连续两次失败才退出”；服务器实现（sha 583bbbcb）
   为一次未达标即退出 + cooldown 2 轮。判为文档错误，按实现披露，不重写历史登记。
3. **bootstrap 登记差异**：本地 Week10–11 YAML 记 bootstrap_seed=20250814，服务器实际执行
   rng=20250722（B=10000，example-level）。以服务器实际脚本与输出为准，差异如实披露，不补写。
4. **reference 覆盖缺口**：safe_reference_random_v2.json 覆盖 17/18 技能；
   math.algebra 无 probe 数据、无下界，三门难度臂永久不可选。属设计限制，
   本周不重算阈值、不重训。
5. **重算 CI 端点差异**：见 week12_claim_revision.md 统计口径披露（~0.05pp，MC 实现差异）。
6. **replay 状态字**：正式 run 中 replay_hysteresis.state = legacy_single_threshold（阈值 0.03），
   coverage_state.enabled = False，均为父配置沿用，非本周改动。
7. **新增分析标识**：Day 73 归一化层、gate 紧凑汇总、Day 74 统计为 Week 12 新增分析，
   输入为冻结 run，未改动任何原始文件与历史官方分数。
8. 跨时期主机/软件差异（Day75 发现，来源：各 era eval_results.json 的 pretty_env_info 自记录）：Week 8 四方法的评测在主机 A（AMD EPYC 9754、驱动 595.71.05、transformers 5.14.1、lm-eval 0.4.12），Safe 三 run 在主机 B（Intel Xeon 8481C、驱动 580.76.05、transformers 4.55.2）。GPU 型号（RTX 4090 D）、torch（2.6.0+cu124）、CUDA（12.4）一致；训练主机按同时期推断但无直接记录。非 GPU 硬件与 transformers 版本差异的影响未知，列入论文限制节。缓解证据：两 era 的 gsm8k 任务配置逐键比对（见 server_schema_notes.md Day75 closeout）、Week 12 归一化 t4（doc_hash 跨方法一致性）通过。
9. 两 era 评测 fewshot_seed 不同（Day75：week8 eval_results.json config 记录 fewshot_seed=20250719；Safe eval_console.log 记录 fewshot manual seed=1234，启动时未显式传入）→ 5-shot 示例集在两 era 间可能不同。生成侧为贪心解码（do_sample=false，console 警告 temperature/top_p/top_k 被忽略），解码确定性不受影响；任务配置逐键比对全 EQUAL（num_fewshot/generation_kwargs/filter_list/repeats/doc_to_text，task version 3.0）。影响未知，列入限制节。
（8 补充）lm-eval 版本跨 era 亦不同：week8 = 0.4.12（eval_results.json 自记录），Safe era = 0.4.8（.venv pip show，venv 建于 9/21 21:09、评测 9/22 01:46、无升级证据，来源等级为推断）。缓解证据：gsm8k task version 双 era 均 3.0，任务配置五键逐键 EQUAL（num_fewshot/generation_kwargs/filter_list/repeats/doc_to_text）。
