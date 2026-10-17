# Q_B全局256个N128任务：只读判断

**结论：当前24核动态版本未找到同方案设备失败，数学/别名/consumer获取没有新阻塞，可推进一次完整CPU核对最小改动。** 不能称全局N128从未做过：旧20核/T96版本已经CPU通过，但没有形成预期Right双槽，未上设备。本轮仅写报告，没有候选、编译或NPU操作。

## 精确历史区别

`C/tests/pypto_test/csa_multishape_spmd_20261016/qproj_n128.py:25–56` 把round上界128改256、列偏移步长256改128，并验证256列块恰好覆盖一次；这是按20个worker全局重分。`HC_MIX_QPROJ_EXPERIMENTS.md:37` 和 `qproj_n128_codegen_audit.json` 记录完整CPU PASS、四个Right都为地址0、`two_right_buffers_present=false`、未提交NPU。不能将“stage2可放下”直接推为编译器必定双槽。

`parallel_20261018/qb_n128_l0_readonly/README.md:41` 则明确保留原N256 owner、内部两条N128链及QR+B满L1常驻；旧 `QPROJ_L1_COLUMNS_EXPERIMENT.md` 同样是原owner内部双列，设备弱改善0.36%未采用。二者都不是本次从正式668直接改全局任务粒度的设备消融。已检索MASTER、VALIDATION_LOG、ATTEMPTS及现有QB专项记录，没有把这几种历史混为同一个失败。

其中 `C=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`；`parallel_20261018` 位于 `C/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/`。

## 正式668的两种M波次和真实owner

`query_projection_stages.py:77–81,132–154`：N块n归 `n%24`，每个N块owner遍历全部动态M128行块；K固定0/256/512/768。设M=`ceil(T/128)`。N256时n=0..127，核0..7各6块、8..23各5块；N128时n=0..255，核0..15各11块、16..23各10块。不是把N/M联合重新分配。

| 工作量 | N256 | 全局N128 |
| --- | --- | --- |
| 0<T≤128，每核MMA（重/轻） | 24 / 20 | 44 / 40 |
| 同区间，每核FIX输出块（重/轻） | 6 / 5 | 11 / 10 |
| 128<T≤256，每核MMA（重/轻） | 48 / 40 | 88 / 80 |
| 同区间，每核FIX输出块（重/轻） | 12 / 10 | 22 / 20 |
| 最忙核负责输出列数 | 1536 | 1408（−8.33%） |
| 全局MMA / FIX次数 | 512M / 128M | 1024M / 256M |

一M有效行是T；两M分别为128、T−128，不把pad128当有效数学。每核有效点积为`所负责列数×T×1024`。核0..7工作量降低8.33%，原轻核8..15反而增加10%，核16..23不变；最忙/最轻列数比从1.2降为1.1。**这是有效工作量，不是QB或整层时延上界。** T更大仍由原动态M循环处理，本文没有验证三M以上设备边界。

权重总逻辑载荷保持`32MiB×M`；QR总逻辑载荷从`128×T×1024B`变为两倍，输出INT32总字节不变。例：T96的一M，权重32MiB不变、QR12→24MiB；T192的两M，权重64MiB不变、QR24→48MiB。对应权重+QR请求均增加27.27%；不等于HBM流量，因为未测缓存命中。每个M/N块的四次QR/权重供数、MMA及一次FIX数量都随N任务翻倍，Scalar地址/循环和局部同步调用也增多。

## 资源、数学和等待兼容

单块物理M128/K256时，Left仍32KiB；Right64→32KiB，两Right理论恰64KiB；Acc128→64KiB。原stage2若保留两份QR Mat和两份weight Mat，局部包络192→128KiB。没有必要引入QR/B全K常驻、双输出Acc或新GM。全MIX真实分配、首段与pipeline回边生命期、两个Right地址和对应就绪/覆盖事件必须看CPU产物；旧20核全局N128单槽结果是明确风险证据。

K256四段INT8→INT32数学不动，1024项最大绝对和16777216小于INT32范围；改变列分割没有跨列归约。每个head仍由后继DQ按原512列RMS/RoPE处理，不改变浮点算术顺序。M尾的`m_rows`须继续传到load、Acc及store，不能只看声明M128。

权重仍是Native 2D NZ描述符，N128对齐原N16块；q_proj仍为 `[pad128(T),32768] INT32`，QR及q_flat/q_heads别名不变（`binding.py:169–174,238`，`entry.py:31–32`）。新每个N128写512B/有效行，列块互不重叠且按缓存线对齐，不新增跨owner写同一行片的冲突。

原head h来自两个N256 owner，现来自四个N128 owner `(4h+j)%24,j=0..3`。`cube_schedule.py:313–314`仍在本核全部N/M结束后发布QB；`vector_schedule.py:761`现全24 AIC获取因此仍安全。原count128与新count256都会令`wait_row_owners`的owner_count=`min(count,24)=24`，但描述应更新为导入 `QPROJ_N_BLOCKS`，不能留陈旧任务数再据此推导窄等待。QR/RoPE等待、DQ owner及所有发布/缓存维护不变，无新增等待环。

## 最小建议与停止边界

只把NZ的 `QPROJ_MM_N_TILE=256` 改128，ND保留512，派生 `QPROJ_N_BLOCKS` 自动成为256；Vector导入该派生常量用于QB等待并修正中文合同。不改M128、四K256、stage2、缓存策略、根Tensor或其它阶段，不叠加已否决的驻留/手工预取。

这足够形成一个明确、可解释的一次CPU候选：核对256列覆盖、动态valid尾、NZ切片、真实Left/Right/Acc/Mat、pipeline局部事件和非QB数学不变。Right是否双槽是检查结果，不是事先通过门槛；若仍单槽，就据实际生成物记录，不追加手工变体扫参。后续是否值得设备对照由主会话依据CPU证据决定，当前不预记收益。
