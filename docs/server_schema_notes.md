# Week 12 服务器字段对应关系（Day 73 取证 + 适配层核验）

## GSM8K samples（lm-eval 0.4.8，15 个正式 run 结构一致）
| 含义 | 服务器真实字段 | 备注 |
|---|---|---|
| 任务 | 文件名 samples_gsm8k_*.jsonl + 记录内 task 元数据 | 不靠 doc_id 猜 |
| 题目键 | doc_id（int，归一化时转 str） | 1319 题 |
| 题目内容 hash | doc_hash | 跨方法 15/15 完全一致（t4 通过） |
| filter 名 | filter ∈ {strict-match, flexible-extract} | 每 (doc,filter) 一条记录，全文件 2638 条 |
| 官方指标 | exact_match（float，顶层） | 官方主指标 = results.gsm8k.exact_match,strict-match（15/15 在 5e-5 内复现，t3 通过） |
| 原始生成 | resps[0][0]（resps 为 [[str]]） | 严禁用 filtered_resps 冒充；两 filter 记录 resps 全等（t2，mismatch=0） |
| 抽取结果 | filtered_resps[0]（str 或 null） | flexible 取全文最后一个数字 |
| 答案 | target | 原评测 target，未重定义 |

已知口径事实：flexible-extract 对“答案后续写”的输出会取到后续内容中的数字
（safe 三个 run 的 strict=1,flexible=0 格为 4/30/88，其余方法 ≈0）。
正文主指标用 strict-match；flexible 差异单独报告并附此机制说明。

## selector 日志（lpf_selector_log.jsonl，每轮一条，4 个 run 结构一致）
round / safe_gate_state.{enabled, active_skills, eligible_skills, fallback_random,
per_skill.{observed_learning_progress, reference_lower_bound, qualifies_this_round,
confirm_count, active, cooldown_until}, reference_schema_version} /
replay_active / explored / replay_hysteresis.{state,threshold,replay_fraction_so_far} /
selected_arm / arm_utilities（54 臂）/ skill_state（17 技能）。

## 读取与适配层（sha256 以服务器 week12 目录登记为准）
- 读取器：raw_decode 整文解析（JSONL 记录内含裸换行，splitlines 解析必崩——Week 10 已取证）
- week12_normalize_gsm8k.py：归一化 + §4.4 四项测试（输出 normalized_gsm8k.jsonl、normalized_tests.json、quadrant_counts.json）
- week12_gate_compact.py：门控汇总（输出 gate_compact_summary.json）
- 两脚本均为只读，不改原始 run、不重写历史官方分数
06aa0106053f7accc6923df36fe38a8c3f5667d30e5484eba3a1f9fb12a7c376  /root/autodl-tmp/campus_project/week12_gate_compact.py
d3b5746a6122dedee8f67a8f5bb8fb7dfad38562bd516ecef49a7ab43ea40c76  /root/autodl-tmp/campus_project/week12_normalize_gsm8k.py
429fdabfe54d9b1a764394e493a5f0c04c26c074287b26c1973ae43797f8fbb0  /root/autodl-tmp/campus_project/week12_stats.py
06aa0106053f7accc6923df36fe38a8c3f5667d30e5484eba3a1f9fb12a7c376  /root/autodl-tmp/campus_project/week12_gate_compact.py
d3b5746a6122dedee8f67a8f5bb8fb7dfad38562bd516ecef49a7ab43ea40c76  /root/autodl-tmp/campus_project/week12_normalize_gsm8k.py
429fdabfe54d9b1a764394e493a5f0c04c26c074287b26c1973ae43797f8fbb0  /root/autodl-tmp/campus_project/week12_stats.py
8a5bcd7a1d2d5c44cf7ba045d6cea5a61c24194767887a703ce7c1f06c955fd1  /root/autodl-tmp/campus_project/week12_evidence_index.py
8a5bcd7a1d2d5c44cf7ba045d6cea5a61c24194767887a703ce7c1f06c955fd1  /root/autodl-tmp/campus_project/week12_evidence_index.py
9c4ccf1725b895e5baaf973a23c54831a23c1febdf2507a4f7766534a0b161d0  /root/autodl-tmp/campus_project/week12_main_table.py

## 版本备注
- gate_compact_summary.json：v1（03:57 首次生成，fallback_fraction 全精度）→ v2（04:08 重生成，
  fallback_fraction 舍入 4 位，其余字段逐一相同，REPRODUCE_CHECK=IDENTICAL 于 04:1x 核验）。
  canonical = v2，生成脚本 sha256 见上表。
bb98e31ebb5de106759d3d214e4503ce4072ede07cada4f18ae676de2f859217  /root/autodl-tmp/campus_project/week12_main_table.py
bb98e31ebb5de106759d3d214e4503ce4072ede07cada4f18ae676de2f859217  /root/autodl-tmp/campus_project/week12_main_table.py
fc2f34fa431173330e1251907f00ec777a310092a4f1c46a5bb87b43ecfad9d7  /root/autodl-tmp/campus_project/runs/week11/safe_fallback_seed20250725/command.txt
7e4d1ed33823de3fcd1b4979a5ff6a3afdd1d6656e9a438e0e5d0df7fa3f5ca8  /root/autodl-tmp/campus_project/runs/week11/safe_fallback_seed20250726/command.txt
9199e58bd80f439c7e8980fb6f403533ac39230cbe1fecbe908ead3e0ae83a5f  /root/autodl-tmp/campus_project/runs/week11/safe_fallback_seed20250727/command.txt
# seed identity: config YAML shared, seed via CLI only (command.txt per run), manifest has no seed field

# --- Day75 supplement: actual-loaded code identity (reviewer request) ---
# safe_selector.py: UNIQUE version on disk, sha256=583bbbcfb1a57b534a464ff9c4d2be671a359cc8df80f04cfd326c5560482b1f
#   (week10_11_starter and code_adapted copies identical; matches gate_checks/safe_selector_sha.txt)
# Safe trainer: campus_project/code_adapted/week10_11/scripts/train_lpf_v2.py
#   gate call sites: L643-644 (mode remap), L1003-1012 (SafeGate.from_file), L1102-1127 (allowed_arms/safe_blocks/filtering)
#   launch-time trainer_sha256 (pipeline_meta.json): 0747b864f79b763762b5d146ab0cbf771a167a393e09e37d21f5e7844409a8f1
#   launch-time config_sha256: fdb45035fa43072789c7ce15bfd3ebbe510675f375d653398d4cd69026c2c2dd
# Base trainer (no gate calls, Week 8 four methods): campus_week8_9_starter/scripts/train_lpf_v2.py (diff vs adapted = 54 lines, all gate-related)
# common.py: single copy campus_week2_3_starter/scripts/common.py; imported at train_lpf_v2.py L55
# seed identity: per-run pipeline_meta.json top-level "seed" + command.txt + dir name (3 sources; run_manifest.json has NO seed field)
# eval_manifest.json in week8 runs = RECONSTRUCTED (schema_version eval_manifest_reconstructed_v1, week10_audit); cite with label
0747b864f79b763762b5d146ab0cbf771a167a393e09e37d21f5e7844409a8f1  /root/autodl-tmp/campus_project/code_adapted/week10_11/scripts/train_lpf_v2.py
7a7d4ae8ed5b40b2f5c352ca270f5092756734e8619c7e7f2f8d8f738defc1dd  /root/autodl-tmp/campus_week8_9_starter/scripts/train_lpf_v2.py
ecf12503e2bd394e9e2082eebfe815256b6a8c4ad2c3d3786ab34b0abc81e1f0  /root/autodl-tmp/campus_week2_3_starter/scripts/common.py
# correction/supplement to Day75 block:
# common.py wiring: train_lpf_v2.py -> upstream(week4_5) -> ensure_week23_scripts() -> common(week2_3)
#   shared fns: aggregate_nll, collate_supervised, per_example_answer_nll, load_model_and_tokenizer (common.py L55+)
# Week8 base trainer assignment = INFERENCE (no pipeline_meta in Aug-14 runs; sole copy in starter + config_provenance=original), not launch-time record
# Safe gate params as-run (week11_safe_curriculum.yaml): safe_candidate_mode=lp_only, safe_progress_margin=0.0,
#   safe_confirm_rounds=2, safe_fail_cooldown_rounds=2, safe_reference_file=reports/week10/safe_reference_random_v2.json,
#   parent_config_sha256=dd9e5d6e3671d0c58677c4a974f6dfaa4f1b9f051fb930740c39cf349680b7f0, probe_batch_size=4
# --- Day75 final: probe interval + eval provenance ---
# probe_interval_rounds three eras: deprecated_probeint4 set (Aug14, all 4 methods =4)
#   -> current week8: lp_only_replay & lp_match_replay =1 (corrected), random_sameloop & lpf_sparse =4 (unchanged)
#   -> safe_fallback =1 (inherits LP parent). Cross-check: probe_seconds lp/safe ~345s (~82x4.2s), random/sparse ~88s (~20x4.2s)
# eval config (random_sameloop_seed20250722, all 15 runs identical protocol): Qwen2.5-0.5B-Instruct main, bfloat16, cuda:0,
#   batch_size 32, limit null (full 1319 docs), seeds 20250719, gen_kwargs {} (lm-eval defaults), bootstrap_iters 100000 (lm-eval's own, distinct from Week12 registered bootstrap)
# fewshot count NOT in config block -> sourced from lm-eval gsm8k task default (see configs block when extracted)
# pip_freeze.txt = 0 bytes (EMPTY) -> historical env snapshot does NOT exist; limitation, do not substitute current env
# env record verdict: .venv/pyvenv.cfg dated Sep21 (post-Week8) -> venv cannot represent Aug-14 env either;
#   archive/server_code_20260921_194732 = Sep21 code snapshot (Week11-era corroboration only). Package versions: NOT RECORDED, limitation.
# eval provenance correction (gsm8k configs block, Day75): num_fewshot=5 (task default);
#   generation do_sample=false/temp=0.0, until=["Question:","</s>","<|im_end|>"];
#   strict-match = regex "#### (...)" take_first; flexible-extract = last-number (group_select=-1) take_first
#   -> config-level confirmation of quadrant-10 mechanism (post-answer continuation hurts flexible only)
#   task versions: gsm8k 3.0, ifeval 4.0; task yaml: lm_eval/tasks/gsm8k/gsm8k.yaml
# --- Day75 closeout: environment as self-recorded at eval time (pretty_env_info; historical, not current env) ---
# Week8 era (eval 2026-08-14): AMD EPYC 9754 256T, RTX 4090 D, driver 595.71.05, torch 2.6.0+cu124, CUDA 12.4.131,
#   transformers 5.14.1, lm_eval 0.4.12 (self-recorded; CORRECTS any earlier note saying 0.4.8), numpy 2.1.3
# Safe era (eval 2026-09-22): Intel Xeon 8481C 192T, RTX 4090 D, driver 580.76.05, torch 2.6.0+cu124, CUDA 12.4.131,
#   transformers 4.55.2, lm_eval_version=None in results (from eval_console.log), numpy 2.1.3
# Cross-era identical: GPU model, torch, CUDA. Cross-era DIFFERENT: host CPU, driver, transformers -> protocol_deviations #8
# PEFT version: not recorded at any era -> limitation. n-shot: gsm8k=5, ifeval=0. eval wall: week8 1118s, safe 990s
# Day75 closeout addendum: cross-era gsm8k task config comparison = ALL EQUAL (5 keys + task version 3.0)
# safe eval_console.log: 1860 generate_until requests = 1319 gsm8k + 541 ifeval (matches Week12 normalization counts);
#   greedy decode confirmed (transformers warning: temperature/top_p/top_k ignored under do_sample=false)
# fewshot_seed differs across eras (20250719 vs 1234) -> protocol_deviations #9
# Day75 final addendum: Safe-era lm-eval = 0.4.8 (source: .venv pip show, earliest checkable; venv created Sep21 21:09,
#   eval Sep22 01:46, no upgrade evidence; provenance grade = inference, NOT self-recorded).
#   => lm-eval version ALSO differs across eras (week8 0.4.12 self-recorded vs safe 0.4.8 venv-state) -> protocol_deviations #8 addendum.
#   Mitigation: gsm8k task version 3.0 in both eras; 5 config keys EQUAL.
1d39390ba9fd880e2644de6aa34282086dbbe1643b46e32377b626b944ba0dc3  /root/autodl-tmp/campus_week10_11_starter/scripts/build_safe_reference.py
05ea5ccec25b5629b643277616549b12869782dad30e0088b76c1b1eab5250fa  /root/autodl-tmp/campus_project/reports/week10/v2_aggregate_snapshot.json
aba0e2813270fa5079bad2f64ead70dde3aff506e5a55ea50cfd0f3247582a4e  /root/autodl-tmp/campus_week2_3_starter/data/week3/probes/analysis_probe.jsonl
8f98e283892d0626b0162165c72dd963e1888cd86b67cc8097d013815ec735b5  /root/autodl-tmp/campus_week2_3_starter/data/week3/probes/selector_probe.jsonl
# probe files: analysis_probe.jsonl=102 lines, selector_probe.jsonl=98 lines (week2_3 data/week3/probes; sha256 above)
# --- Day76: reference rebuild verification (reviewer request) ---
# selector_log integrity: manifest selector_log_sha256 vs current file sha256 = 15/15 MATCH
# formula regeneration: build_safe_reference.py + original 3 random_sameloop logs -> /tmp/safe_reference_rebuild.json
#   SEMANTIC MATCH: 17 skills, lower_bound 0 mismatches (tol 1e-9), n=244 all, meta equal (random_sameloop/burn_in=1/z=1.96)
#   byte-level: N/A (absolute paths embedded in run_dirs field, by design)
# load-time reference hash: NOT RECORDED (grep be63f117 over reports/week10/*.json) -> recorded as MISSING, not fabricated
# manifest train_sha256 field: identical value 0896b165... in BOTH eras; equals NEITHER week8 base trainer (7a7d4ae8...)
#   NOR safe adapted trainer (0747b864...) -> field semantics UNEXPLAINED (legacy); NOT citable as trainer identity.
#   Authoritative trainer identity: pipeline_meta trainer_sha256 (Safe, launch-time) / inference from sole copy (Week8)
# week8 base trainer sha256 registered: 7a7d4ae8ed5b40b2f5c352ca270f5092756734e8619c7e7f2f8d8f738defc1dd
