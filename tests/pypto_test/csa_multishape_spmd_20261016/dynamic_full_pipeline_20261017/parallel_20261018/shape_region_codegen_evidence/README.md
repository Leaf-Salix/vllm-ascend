# 整层动态CSA的区间编译候选

2026-10-18。作用正式 `ops/pypto/deepseek_v4_flash_csa_spmd`，新开关
`shape_specialization` 默认关闭；不重做已经失败的4/20核调度或WS转置候选。
目标仍为七档完整L1逐项相对原分task降低30%，本候选尚未取得性能结论。

## 改动和动态边界

这次不改变数学tile选择，而是把原来已由Tensor shape决定的分支选择移入
constexpr签名，让编译器按区间裁掉不会执行的数学路径：

| 事实 | 来源与用途 |
| --- | --- |
| PROJECTION_M | T<128使用原M32；128<=T<=512使用原M64；T>512保留原动态分段。 |
| TOKENS_ALIGNED8 | 仅T被8整除时成立；HC和主Q反量化仍按device T领取行块，但不含不可能执行的短行回读分支。 |
| DQ_HEAD_TILE | 直接复述原48-AIV整波head分组规则，取4/8/16/32/64；不改变原分工结果。 |
| ATTN_KV_ORDER | 直接复述原144行分界的事件顺序；每个有效块原有通知、等待和三槽协议不变。 |

所有T/B/Q/L/位置/页表/compact计数仍是动态描述符或device内容。Host只读取原来
已有的hidden.shape，没有新增item、D2H、meta kernel或逐步tiling传输。不同策略
可以编译不同二进制；没有把B16/S6或全部shape常量写入数学代码。对齐性质不是
“每请求长度相等”，mixed请求和dummy仍由原device metadata处理。

调用前由 `validate_shape_facts` 核对策略与实际T匹配，防止把8对齐二进制用于
未对齐T。每个helper继续读取真实T、计算owner、波次、偏移和有效范围；默认
参数保留原通用路径。未对齐的S1/S5场景没有被删除，新增路径的设备验证尚待完成。

## 已有证据

- `source_reference`：37e78245时原33模块；`source_candidate`：本轮冻结候选。
- `check_shape_facts.py`：直接执行旧DQ函数的Scalar前缀，T=1..4096与新规则一致；
  覆盖128/144/512边界，不是性能成本模型。
- `cpu_long_v1`：测试命令传入相对controls路径，wrapper切换目录后找不到文件；
  未进入编译。这是测试命令错误，不是组件bug。
- `cpu_long_v2`：长历史候选完整CPU/PTOAS通过，130根、58动态轴、单MIX/sync_start，
  不提供样例shape。`cpu_short_v1`：短历史/M64/head32候选同样通过。
- `cpu_reference_v1`：使用相同独立PyPTO重新生成原版，避免跨编译器混比。
- `REGION_CODEGEN_SIZE.json`：原/候选的PTO函数静态规模对照。去掉未执行分支
  不等于运行时少做数学，代码体积、指令site数量也不等于完整L1性能收益。

依赖继续使用独立 PyPTO `csa-spmd-memory-reuse-20261018`（含本地d46a813ca常量
起点修正）和既有Simpler参数扩容分支，未更改共享安装或依赖二进制。

## 七档设备验证

`SEVEN_SUBMISSIONS_V1.json` 保存真实任务ID。每档均为正式第4层W8A8、S6，
原多task／原单SPMD／候选同进程比较，先A→B→A、精确输出/TopK/score、完整
cache/state、保护区、代际与基本图重放，再取五窗完整L1中位数。所有任务经
`task-submit --device auto`，不重测Native或静态参考，也不以提交成功当作PASS。

本节结果待任务终态追加；固定S6通过也不能替代S1/S5、mixed-Q、策略边界验收。

## 终态

七档全部exit0、完整功能检查通过；性能见 [结果表](SEVEN_RESULTS_V1.md)。
候选未形成全区间改善，30%目标0/7。正式十个文件已经恢复，完整候选留在本目录；
不再对这组开关扫参，未追加S1/S5设备用例或重复七档。

实际run_device/check_cpu及其辅助模块已复制到本目录，同级相对目录合同不变。
复现脚本改用本目录冻结驱动；本次已执行的提交命令仍原样保留，不伪造历史路径。
