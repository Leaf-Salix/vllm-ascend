# 交错候选的信号容量与隔离验证入口

2026-10-18。此部分只改候选 `source_candidate/runtime_abi.py` 和 `binding.py`，以及本目录独立 runner。原 `dynamic_full_pipeline_20261017` 脚本、正式 ops、共享安装和旧依赖没有修改。候选的数学 helper / schedule 由其他代理分别维护。

原 `PHASE_COUNT=60` 和0..59阶段含义保持不变。新增常量为 `WAVE_PHASE_BASE=60`、`WAVE_PHASE_STRIDE=3`、`WAVE_ROOT=0`、`WAVE_TOPK=1`、`WAVE_PLAN=2`。第w波的三个独立phase为 `60+3*w+kind`。`signal_phase_count(T)` 只按输入 `x.shape[0]` 返回 `60+3*ceil(T/8)`；`PreparedDynamicCSA.signal_phase_count` 保存实际分配容量。实际波数和边界仍读device group描述符，Host不读Q/L/position，不按B*S实值化，不新增根Tensor或动态轴。

每phase仍是72核×128 INT32，每核只写word0。phase0的generation仍每核每调用推进一次，wave不重编码generation、不复用同代phase。T=0/96/192分别分配60/96/132个phase；没有真实group时新增槽可全不发布。`wave_boundaries` 要求除末波外至少8个真实非空group，每组至少一行，才能使用这个容量上界。

`signal_checks.py` 逐实例核对容量再reshape。reference仍可只有60phase，不能使用candidate容量读取它。`run_device.py` 原三处固定reshape、`output_policy_matrix.py`三处、metadata和padding replay各两处均已替换。eager、graph/L1后的原phase0代际检查、严格0容差数值比较、Native/scratch保护区、metadata同址图重放、padding有效前缀检查均保留。

原输出owner检查仍接收 `observed_phase_slots` 的60phase前缀，原 `check_output_phase_signals.py` 逐字未改。新增 `observed_wave_phase_slots` 单独保存新增phase word0，`signal_padding` 分别报告旧/新所有word1..127是否为零。新增检查没有放开padding；当前私有入口明确拒绝 `--phase-clock`，以免通用时钟hook写入wave保留位。wave word0可能保留过去代际的未复用槽，因此padding检查不把所有预留wave槽一律要求为当前代际；实际发布者/等待关系仍需独立协议和设备验证。

`prepare_runners.py` 记录原脚本来源并生成有限差异 `RUNNER_CHANGES.diff`。`RUNNER_PREPARATION.json` 保存来源路径与大小。`check_runner_contract.py` 只用标准库，读取实际容量函数AST，检查T=0..4096、旧reference容量、5个新旧padding污染反例、容量不匹配反例和脚本语法；结果 `RUNNER_CONTRACT_CHECK.json` 为PASS。这里的稀疏Signal替身只验证检查函数会拒绝污染，不是设备协议或Torch张量运行验证。

两个入口都使用已经验证的Python overlay：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_parameter_overlay_20261018/pypto_overlay`

它保留原隔离依赖源码修复、只增加Inline Tile注解保留；扩展 `.so` 只链接原隔离依赖。wrapper设 `CSA_STATIC_PYPTO_IMPORT_MODE=filesystem`，device runner分别核对Python来源和扩展符号链接解析目标；没有放松成任意来源均可。原formal33源reference在 `source_reference`，候选及其新模块在 `source_candidate`，各自独立包与policy。

两个新wrapper均在source环境前 `cd "$repo_dir"`。主会话首次从workspace根启动v2时，既有环境激活因cwd立即exit128、日志为空且未创建输出目录；这不是CSA编译失败。从仓库cwd启动后正常进入编译。此修正只改本目录wrapper，不改原env；后续v2由主会话接手，本代理未重复启动。

已授权并且只执行一次完整long CPU：

```bash
bash reproduce_cpu.sh <候选根>/cpu_long_v1 --strategy long --attention-query-metadata off
```

控制为G6/N64/score1024/prefetch1/toL0/balanced/roots1；不传样例shape。`cpu_long_v1/result.json` 显示 symbolic parse通过、动态Tensor轴仍在；Inline pass失败：

```text
AssignStmt var is not a Var after mutation
source_candidate/attention_tick_stages.py:208:13
mutator.cpp:632
```

该行是 `qk_q = pl.gather_row(qk_q, ...)`。原失败source快照和所有日志留在 `cpu_long_v1` / `cpu_long_v1.log`。已立即报主会话，由该数学源负责人处理实际TupleGetItem替换到赋值LHS的规避；本代理没有修改该文件或重新运行整层CPU。小Mat/Left smoke的两次独立结果见 `mat_left_smoke_evidence/README.md`，其v2通过不能代替整层通过。

设备入口已准备，尚未执行。必须由主会话在CPU和源码审查完成后经 `task-submit --device auto` 调用本目录 `run_device.sh`；保留原 `--history/--batch/--wait-backoff/--metadata-replay/--padding-replay/--source-matrix-reference/--l1-windows` 入口。不要绕过队列裸跑，也不要以此报告宣称设备正确性或性能改善。

主会话后续v2进入SSA后报228条错误，原622383862字节result与311244311字节日志原样保留。为避免重复打印每条诊断附带的完整大函数，私有 `check_cpu.py` 改用 `cpu_failure_report.py`：完整异常另存 `failure_error_full.txt`，traceback只存栈；result/stdout限4096字符错误预览，Message/Location各最多64种且每种最多512字符，仍保存完整文件路径/大小。`prepare_runners.py` 同步此变换。标准库合成检查 `CPU_FAILURE_REPORT_CHECK.json` 为PASS，不涉及新编译或放宽任何检查。

最小作用域对照及完整失败历程归档于 `unroll_iterarg_evidence/README.md`：直接IterArg经过unroll失败，调用前fresh Scalar alias通过SSA。未修改compiler，原组件缺陷仍在。主会话v3包含alias及独立while/if修正，已通过SSA但因Left77824>65536在内存分配阶段失败；资源诊断由其他代理处理，此处不重复编译。
