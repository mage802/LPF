# CAMPUS Reproduction Pack

When Competence-Aware Curriculum Fails: strict token/update auditing of CAMPUS scheduling and low-cost diagnosis.

本仓库提供该研究的完整复现材料：训练代码、冻结配置、数据构建脚本、探针、运行日志/账本、评测原始输出、审计脚本与 Week 12 证据链文档。

## 目录结构

```
code/
  training/
    base/train_lpf_v2.py        # Week 8 四方法训练器（无门控）
    safe/train_lpf_v2.py        # Safe 训练器（含 SafeGate 调用，L643/1003-1012/1102-1127）
    safe/safe_selector.py       # SafeGate 实现
    common.py                   # 共享更新函数（掩码/token 计数/模型加载）
    upstream.py, gradient_sketch.py
    build_safe_reference.py     # 门控阈值生成器
    safe/apply_safe_gate_patch.py, build_week11_candidate_config.py
  data_build/                   # 数据采样/技能标注/难度分桶/probe 构建脚本
  analysis/                     # Week 12 全部复算脚本（gate/stats/evidence/table/efficiency/bootstrap/审计）
configs/
  week11_safe_curriculum.yaml   # Safe 共享配置（三个 run 共用，seed 经 CLI 覆盖）
  week8_lp_only_replay.canonical.yaml  # 父配置
  runs/<run>/config.yaml        # 15 个正式运行的冻结配置（Safe 另有 config_snapshot/command.txt/pipeline_meta.json）
  static/<run>/                 # 6 次 Static 正反向运行配置与 manifest
probes/
  selector_probe.jsonl          # 98 条
  analysis_probe.jsonl          # 102 条
  safe_reference_random_v2.json # 17 技能门控阈值（burn_in=1, z=1.96, n=244/skill）
  safe_reference_random_v2.sha256
  v2_aggregate_snapshot.json    # 阈值生成输入快照
manifests/
  runs/<run>/run_manifest.json  # 15 个运行账本（含五工件 sha256、timing_seconds、预算账本）
  data/                         # 数据构建 manifest（week1 数据量/哈希、技能划分、bucket 划分）
logs/
  runs/<run>/                   # lpf_selector_log.jsonl（选臂记录）、train_log.jsonl、Safe 另有 eval_console.log
eval/
  runs/<run>/eval_results.json, eval_manifest.json
  samples/<run>/                # lm-eval 逐题 samples（gsm8k 2638 条/run = 1319 题 × 2 filters；ifeval）
  normalized_gsm8k.jsonl        # Week 12 规范化样本（bootstrap 原料，15 run × 2638 条）
  normalized_tests.json, quadrant_counts.json
audit/
  week10/                       # Week 10/11 审计现场（preflight、config 继承审计、artifact/samples 审计）
  week12_gate_checks/           # 门控机制核验（4 run gate JSON + verdict + 合成单测）
  evidence_index.json           # 15 run 身份索引
docs/
  protocol_deviations.md        # 9 条协议偏差披露（含跨 era 主机/软件差异）
  server_schema_notes.md        # 字段映射、全部登记 sha256
  week12_claim_revision.md      # 主张修订表
  week12_exit_checklist.md      # 放行清单（Ready for manuscript）
  week12_main_table.md, week12_efficiency_15runs.md, week12_main_stats.json, week12_per_seed.md
SHA256SUMS.txt                  # 本仓库全部文件的 sha256 清单
```

## 复现顺序

1. 环境：见下方"环境与已知限制"
2. 数据构建：`code/data_build/`（采样 → 技能标注 → 难度分桶 → probe 构建）；产物校验对照 `manifests/data/`
3. 训练：`python code/training/<base|safe>/train_lpf_v2.py --config configs/... --seed <seed> --output-dir <dir>`
   - Safe 三个 run 共用同一 config，seed 仅经 CLI 覆盖（`configs/runs/safe_*/command.txt` 为证）
4. 评测：lm-eval gsm8k + ifeval（5-shot / 0-shot，贪心解码；调用见 Safe run 的 `logs/.../eval_console.log`）
5. 复算：`code/analysis/` 全部脚本从冻结输入（manifests/logs/eval）重算 Week 12 全部结论

## 环境与已知限制（如实披露，勿回填）

- GPU: NVIDIA RTX 4090 D（两 era 一致）
- Week 8 era（2026-08）: torch 2.6.0+cu124, CUDA 12.4, transformers 5.14.1, lm-eval 0.4.12, numpy 2.1.3, host AMD EPYC 9754
- Safe era（2026-09）: torch 2.6.0+cu124, CUDA 12.4, transformers 4.55.2, lm-eval 0.4.8（venv 状态，推断级）, host Intel Xeon 8481C
- 跨 era 差异（主机 CPU、驱动、transformers、lm-eval 版本、fewshot_seed）已立案：见 `docs/protocol_deviations.md` #8/#9
- PEFT 版本：未记录（缺失，不补造）
- Week 8 训练器归属：推断级（无 pipeline_meta）；Safe 训练器身份以 `pipeline_meta.json` 启动时 trainer_sha256 为准
- 阈值 JSON 内嵌绝对路径：字节级 hash 不比对，语义级重建已验证（17 技能 lower_bound 零不符，容差 1e-9）

## 模型权重

15 个最终 LoRA adapter（各约 34MB）不在本仓库；如需请通过 <链接占位> 获取。sha256 清单见各 run `manifests/`。

## 许可

见 LICENSE。
