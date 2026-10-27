# 分段差距与QA seed实验复现证据

## 身份与状态

正式运行控制11b380e3f，分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。CPU/单层新SDK为PyPTO d44925c1/PTOAS914863ed/Simpler57e499；whole候选仍用原已验2dfa7f12/0.66，不混为同一资格。

- `executed_helpers.tar.gz`封存已编译/运行的harness、SOURCE及manifest，包含已否决WKV/HCsplit、QA硬删除v2、保留正式95及atomic保护的v3、poison单元、whole候选、Indexer NZ独立候选；各文件身份由各自编译收据及result决定。
- `generated_cpu_evidence.tar.gz`保留实际CPU生成的CPP/PTO/metadata/receipt/编排。去掉重复passes_dump及.so/.o二进制；`RAW_CPU_INPUTS.json`记录本地完整原始归档SHA。保留binary_manifest中的原记录SHA，未修改生成C++。
- 其余raw tar为各任务的result/log/DFX。`qa-poison-result.json`及log是9形状27场景的完整物理输出overwrite检查，gold为原QA；不是Native整层或token。
- `NEW_SDK_STAGE_ANALYSIS.json`及脚本为新依赖图与独立8窗L1的严格映射；所有72核、单replay、无丢失、每task都有依赖映射，CPU/JIT binary map相符。
- `QA_NO_SEED_V1_REJECTION.md`记录从错误formal路线派生的候选在设备前否决；最终测试采用v2；正式guard候选又单独保留95路线。不能混用3个硬删除seed/60共同leaf与4个if0 seed/65相同leaf两套结论。
- `QA_WHOLE_DERIVATION.md`说明whole的当前CPU收据与继承旧字段。历史FORMAL_CPU_RECEIPT/SOURCE谱系不当作本次设备证据。
- `QUERY_ISLAND_NEXT_PLAN.md`是后续较大候选的审查方案，尚无设备性能/精度结论，不计入已取得优化。

`qa-guard-8k-run1.tar.gz`和`qa-guard-8k-reverse.tar.gz`是最终保留atomic/95路线的两次复测，不能被硬删除v2结果覆盖。

## 分段数据的本机重算

在新目录复制本目录的分析脚本，按下列结构解压即可用标准Python重算，无需NPU：

```bash
mkdir -p collected/direct-dependency collected/cumulative-dfx
tar -xzf direct-dependency-evidence.tar.gz -C collected/direct-dependency
tar -xzf cumulative-dfx-evidence.tar.gz -C collected/cumulative-dfx
tar -xzf generated_cpu_evidence.tar.gz -C collected cumulative-jit-map
python analyze_stage_cuts.py
```

输出应与封存JSON一致，不能用旧SDK映射替代新SDK或把相互重叠的函数包络相加成纯计算。nalinaly两段来自独立cut，和原kernel内部差值有编译布局影响，不是可直接节省的时长。

## NPU重跑顺序

1. 新自有目录解压helpers，核对manifest及SOURCE每文件SHA，恢复固定canonical源码/SDK/bank/gold；若路径或源码变化，重新封存manifest并CPU重编，不复用旧收据。helper的source_identity/sdk_bind会拒绝混用源或工具链。
2. 两臂分别`bash compile.sh baseline`和`bash compile.sh candidate`。whole为`bash compile.sh tuned`；poison为其独立`compile.sh`。须root64（poison3）、artifact.load、NPUfalse。显式checkout ccache与四并行已在env中。
3. 单层一任务一配置，外780/内750秒，不在登录shell占卡：

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash <helper>/run.sh <new-label> <batch> <history> <comma-separated-query-lengths>"
```

8K/B32为history8192、32个6；128K/B16为131072、16个6；T95为15个6加5；混长126为1～6重复6次。独立反向捕获重复加`--reverse-capture`。已证明主配置性能收益后的边界回归可加`--qualify-boundary`，result会明确记录这是精度回归，不冒充该形状性能筛选胜出。
4. 每项先event12×100与单独L1；event中位至少改善0.5%、12对至少10对领先、L1更低，才验Native A/B八边界、ABA、guard/只读。顶层pass只是流程完成，必须结合winners与accuracy_executed，不能把未做精度说成通过。Native deterministic level1、HCCLtrue、NZ2、TP1、static13包、SK0。无DFX普通计时与L1/dep分别运行，不删离群值。
5. QA单元的`qa-poison-v1/run.sh <new-label>`只验QA overwrite；原权重key、文件、SHA在JSON中。eager/raw FULL、NaN/Inf/finite poison、完整ceil16输出及guard、每phase输入只读、ABA全部需要通过。
6. 稳定收益及单层精度通过后，whole只验CSA替换，不改HCA/FFN。`whole/run.sh 8k`通过再`whole/run.sh 128k`，16卡、warm192+测量192，复用固定Native gold，不跑Native2。8K40活跃/T240/22000块；128K24提交21活跃/T126桶144/24000块/ring56,96,176,1MiB/window2048；zero-preempt、实际入场、16rank token与接受事件及PTO/Native计数增量都由原audit检查。每任务仍780/750秒。

最终guarded v3反向复测仅降低0.1624%，低于预设0.5%门槛，不采用。原已排队whole8K尚未获卡，已取消；没有本轮新候选的整模型token资格，不以历史whole结果代替。
