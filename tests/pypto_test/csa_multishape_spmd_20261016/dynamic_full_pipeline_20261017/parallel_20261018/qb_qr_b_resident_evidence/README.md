# QR 与 B 各 256 KiB 常驻：独立 CPU 候选

2026-10-18。本目录从已冻结 B-only 候选派生，仅修改 `source_candidate/query_projection_stages.py` 的 `QB_B_RESIDENT=True` 分支和中文说明。正式源码、B-only 源与已有结果均未修改。唯一完整符号 CPU 已 PASS：130根、58轴；真实 QR/B Mat 为0/262144，恰512 KiB，Right单槽覆盖等待保留。没有设备执行、精度或性能结论。

三方参照必须分开：正式 `66882634` 使用原 M128/N256/K256 streaming；B-only 在每个 N256 内保留整 K 权重、QR K256流水；本候选在每个 AIC 的 N 循环外保留完整 QR，在 N 内保留 B。B-only 原 v1 的 Right131072 超容量失败仍保留在原目录，不能记作本候选结果。

## 实现与边界

Host 原有 `128<T≤256` 策略及 constexpr 接线逐字继承 B-only，不新增开关。True 支载入 QR INT8容量 `[256,1024]`、有效 `[T,1024]`，每核一次；每 N 载入 B INT8 `[1024,256]`，跨两个 M128复用。两个 Mat 各262144 B，共524288 B。False 支 AST 与 B-only 完全相同，源级其他函数不改；没有为 False 重复编译。

M 循环仍按设备动态 `ceil(T/128)`，T129/255/256分别是有效行128+1、128+127、128+128。每 M 的首 K0用 `tile.matmul` 建 INT32 Acc，普通 K循环依次256/512/768做 `matmul_acc`。本机正式实际生成物原本就是四 K256，未新增 K归约分块变化。与 B-only 的区别之一是去掉该开启分支的 `pl.pipeline`：每循环体只有一份 Left/Right extract与一次MMA，避免 IO重排将两份64KiB Right同时置于首MMA之前。

`tile.extract` 的本地类型推导（依赖 `src/ir/op/tile_ops/transform.cpp:784–835`）会用源动态valid减窗口偏移、裁到128，并对部分 Mat→Left/Right结果选择 CompactNormal；无需把固定capacity当有效T或手工补零。最终仍须检查生成的 Left/Acc/store有效行。Right一次64KiB占满L0B，覆盖等待必须由后端保留；不改任何通知、FFTS或外层cache/fence。

理论上复用 B 可把双M的逻辑权重GM请求64→32MiB；QR请求从 `128×T×1024` 减至 `24×T×1024`，例如T192是24→4.5MiB。这不是 HBM 流量或性能保证。满512KiB同时也没有备用Mat槽，编译器新增物化/保守活跃都可能导致失败；不因理论能装下就放宽实际容量检查。

## 唯一 CPU 命令

完整命令保存在 `run_cpu.sh`，拒绝覆写已有 `cpu_v1`：

```bash
bash /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qb_qr_b_resident_cpu_20261018/run_cpu.sh
```

其调用主 `D/reproduce_cpu.sh`，参数为 `--attention-query-metadata off --spmd-source source_candidate --controls-json controls.json`，输出 `cpu_v1/`、stdout/stderr `cpu_v1.stdout.log`。controls与冻结B-only直接文本相同：G6/panel64/score1024/prefetch1/OA256/OB128，`QB_B_RESIDENT=true`，T/B及其余58轴仍符号，不提供真实样例shape。

源级审计与差异在 `source_audit.json`、`candidate_vs_b_only.patch`，生成脚本为 `make_candidate.py`。候选33源自启动本次CPU起冻结。本目录不计算hash，不复制大历史产物，不启动NPU。

## 实际生成结果

`cpu_v1/result.json` 的 `passed/full_codegen_passed/single_mix/sync_start` 均为true；样例shape未提供，设备执行/精度/性能均为false。只执行一次 CPU，没有失败修正或重编。

`generated_audit.json` 记录全部空间实际地址末端，`qb_events.txt` 是带原行号的Q_B操作/事件摘录；可由 `audit_generated.py` 读取现成产物复核。该脚本还确认将False分支展开后的函数体AST与冻结正式668相同，不声称False已经另行CPU或设备验收。

| 空间 | 全函数最大地址末端 B |
| --- | ---: |
| AIC Mat | 524288 |
| AIC Left | 65536 |
| AIC Right | 65536 |
| AIC Acc | 131072 |
| AIC Vec | 16384 |
| AIV Vec | 163872 |

Q_B实际Mat仅两份：`cpu_v1/kernel/ptoas/dynamic_csa_full.pto:1476` 的QR `[256,1024]` 在0，`:1486` 的B `[1024,256]` 在262144。没有中间Mat物化；两者保持跨N/跨M的预期生命周期。Left/Right/Acc均在各自空间地址0。PTO `:1492–1520` 使用 `min(max(T−t0,0),128)` 贯穿compact Left、Acc和store；`:1504` 的K循环仍256/512/768，INT32初始化与累加顺序不变。B load 的 `l2_bypass` IR属性保留，不据属性声称最终指令的实际缓存行为。

实际C++ `cpu_v1/kernel/kernels/aic/cube_schedule.cpp`：

- QR仅 `:10367` load一次，在N循环外；每N的B仅 `:10408` load一次，进入两M之前完成 MTE2→MTE1 acquire。
- `:10466` 等 M→MTE1 event1，再 `:10469/:10479` 提取Left/Right；`:10482` MMA后归还event1。下一个K不会提前覆盖仍在消费的单64KiB Right。首K与循环交接还有`:10455–10457`的set/wait/set，未擅自删除。
- 每个M的首次MMA在`:10453`等 FIX→M，避免前M输出store未读完就复用唯一128KiB Acc。
- N回边 `:10525` 等 M→MTE1，`:10526` 发 MTE1→MTE2；下一N B load先于`:10407`获取该许可。跨N覆写有保护，但其链仍等待上一MMA，不能宣称下一B搬运已经与上一最后MMA重叠；与FIX实际重叠量也未测。
- 末尾`:10532–10556`保留原pipe drain、DCCI、DSB与QB通知。没有新增跨核通信或删业务fence。

这些是生成和容量检查；未逐一声称非QB所有分配/事件与正式668相同。设备对照参考应为原F的 `source_reference/`（正式668），不能使用失败的B-only v1。源及CPU产物现已冻结；`manifest.json` 列可归档文本和本地大产物路径/大小，不复制binary。

## 非QB分配的有界对照

另用刚完成并冻结的 **B-only v2** `F/cpu_v2/kernel/ptoas/dynamic_csa_full.pto` 作严格同控比较：两边`result.json`的controls完全相同，均OA256/OB128、G6及QB开关True，33源仅`query_projection_stages.py`不同。结果在`non_qb_allocation_comparison.json`。按真实源函数范围排除`_q_b_nz_stage`后，比较其余allocation的生成顺序及`空间/dtype/rows/cols/address`，不靠SSA名字或位移后的源码行号配对。

535项非QB AIC allocation中262项改变地址，全部是Mat，shape/dtype/容量均未变；其余273项一致。其中259项Mat下移64KiB，3项Attention query Mat由0改为393216。这对应满512KiB方案引起的全局Mat排布变化，不能称“只有QB物理地址变化”。1812项AIV allocation逐项相同。没有据此推断性能，也没有为检查额外编译。

JSON附带另一份旧9d设备PTO的同样比较，其数学分支与该区间等价；但它不是新668 CPU，严格同控结论采用上面的B-only v2，未将旧9d产物充当正式668新生成。非QB完整指令与事件尚未逐项比较；本次仅证实际地址/容量和QB事件，整链后果由正式668的同轮设备对照判断。

`qb_pto.txt`只摘录当前PTO的QB区段，保留原行号与指令，去掉末尾源路径loc；`qb_events.txt`保留C++操作和事件原行号。大PTO/C++留在本地原路径，未复制到其它工作树。本报告未对尚在执行的两档设备任务作精度或性能结论，也没有追加任务。
