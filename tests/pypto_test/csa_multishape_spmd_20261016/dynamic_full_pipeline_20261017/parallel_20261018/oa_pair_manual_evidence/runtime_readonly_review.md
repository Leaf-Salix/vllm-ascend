# OA pair 正式迁移前的独立只读审阅

审阅对象：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate` 的四个修改文件。对照该目录的 `candidate.patch`、`source_audit.json`、`interface_ast_audit.json`、`CPU_REVIEW.json` 与 `owner_evidence/README.md`。本轮只读源码及已有证据；未修改冻结源、正式源，未编译、上卡或计算 hash。只新增本报告。

**结论：在现行固定 24 AIC / 48 AIV / 72 物理核、原 NZ 权重与容量合同下，未发现需要修改运行代码的 owner、等待、padding 或动态尾行问题。** 需要修正的是旧模块总览对分工和显式 L0 路径的描述。主会话仍在收集其余目标与 S1/S5，本审阅不替代那些结果，也不据首两档结果扩大性能结论。

## 运行合同核对

| 项目 | 具体位置与结论 |
| --- | --- |
| 同谓词 | `cube_schedule.py:403`、`vector_schedule.py:925` 都在 `OUTPUT_GROUP_PIPELINE` 内，再以 `OA_COLS == 128 and 0 < tokens <= 96` 选择 pair 生产/五 owner 等待。T=0、T>96、N256、关闭组流水均保留原路径。 |
| 唯一生产者 | `output_projection_a_stages.py:270`：组 g 的 pair p=0..2 归 3g+p，写 N128 列块 2p、2p+1；single s=0..1 归 (3g+3+s)%24，写列块 6+s。组内五 owner 互异，组 7 的 single 正确回绕到核 0/1；八组共 48 个 pair 列块加 16 个 single 列块，恰好覆盖全部 64 块。每核一个 pair，另有 16 核各一个 single；同一核跨组的写入使用不同 phase。 |
| 等待覆盖 | `cube_schedule.py:409` 的领取条件与 helper、`phase_sync.py:159` 的五 owner 公式一致。消费者只在自身有量化任务的组调用 wait，但每次获得该组完整 1024 列。现行 `runtime_abi.py` 固定 GROUPS=24，故生产侧 %GROUPS/%WORKERS 与消费侧 %24 等价；此 helper 不能推广到任意 G。 |
| 无新等待环 | 每个 AIC 先完成自身所有组的 O_A，才进入原 O_B 量化等待。O_A helper 内不等量化，AIV 量化仅等 O_A，再发布给 O_B；没有“量化等该核未来 O_A、该核先等量化”的环。新分工会改变各组就绪先后，不能描述为保留原每核任务序，也不能保证每组都更早。 |
| 发布获取 | 每个实际 owner 完成本组全部写入后只发布一次。原 `publish_phase` 的流水/缓存维护保留；新 wait 收齐五个同代信号后执行原样 fence/cacheinvalid/fence。72 核 begin、代际与 512 B 单写者信号跨度不变。T 或策略切换后，仅等待本代实际 writer；不需要清空不活跃槽，也不读取其旧代值。 |
| 动态有效行 | pair 只有一个 M128 容量块，因为实际 T≤96；packed 组步长仍取真实容量 / 8，T 不由容量反推。两条 Acc 经 SetValidShape(T,128)，load 与两个 store 保留 T 有效行；single 保留原 valid_shape→matmul/acc→assemble。既有 CPU 审计确认两个 store 均为 [T,128]，不能把 M128 容量理解为可向不足 128 行的 GM 写满 128 行。 |
| padding 与 O_B | `output_projection_a_stages.py:189` 量化 helper 未改：仅读取有效 FP32 行，每组最后活动 32 行任务负责 INT8 的 [T,ceil(T/OB_ROWS)*OB_ROWS) 尾零，不读取 FP32 padding，不补写 scale padding。`vector_schedule.py:950` 在尾零完成后发布，`cube_schedule.py:426` 后的 O_B 仍等该组全部量化 owner，故尾零的 happens-before 保留。 |
| 数学与片上生存期 | pair 两条 FP32 Acc 各自按 K128 升序累加，首段才 init；共享 Left 并未合并两条归约。single 与所有量化函数保留原实现。已有 CPU 审计记录 Mat 全局上界 475136 B、Left/Right 各 65536 B、Acc 131072 B；本轮未重新编译或自行重做生成物审计。 |

`owner_evidence/README.md` 前部保存的 `single=2g+s` 是已放弃的历史模型；本报告与冻结代码采用其后部最终的 `single=(3g+3+s)%24`，不可从旧段落复制映射。

## 建议替换的中文说明

以下是供主会话迁移时使用的文字，本轮没有应用到源码。模块总览和函数说明当前是 Python **docstring**：修改它们会改变完整 AST 中的字符串常量。若迁移检查要求“不改执行 AST”，应明确只忽略 module/function docstring 后比较其余节点；严格完整 AST 不变只能修改 `#` 注释，不能同时宣称 docstring 已替换且完整 AST 一致。

### output_projection_a_stages.py

模块开头“外层显式分核的三个核内阶段”不再准确；第 28–31 行无条件描述 `logical%WORKERS` 和“不搬显式 L0”，与新路径直接矛盾。建议把“输入输出与阅读顺序”的投影部分及“分工、数学与尾行”的投影部分改为：

> 动态 CSA 的 O_A 分组投影与分组量化。
>
> AIC 消费 attention 的 packed heads，按八个 O group 独立做 K4096→N1024，写 FP32[P_MM,8192]。开启输出组流水、OA_COLS=128 且设备实际 0<T≤96 时，使用 o_projection_a_pair_group_stage_nz；其它区间保留原独立列块 helper。量化仍对每行每组 1024 列独立取 amax，写 INT8 与 FP32 scale，不跨八组归约。
>
> 原路径以 (group,ceil(T/128) 行块,N 列块) 展平并按 logical%WORKERS 分给 AIC。小 T pair 路径专用于 24 AIC：组 g 的三个 pair 分给 3g..3g+2，每核共用输入和 Left，分别累加两个 N128 输出；最后两个 N128 块分给 (3g+3+s)%24。pair 使用显式 Mat/Left/Right 与两个 FP32 Acc，每条链按 K128 顺序累加；single 保留原 K256 首块和后续流水。T 和 GM 容量保持动态，M128 是 tile 容量，读写仅覆盖实际 T 行。

保留现有量化数学、容量与尾零说明。将“同步与生命周期”第 39–43 行替换为：

> 原 groups helper 保留 logical%WORKERS；pair helper 使用与 cube/vector 同步的五 owner 映射。两种 helper 均不创建任务或信号。每个 AIC 完成本核全部 O_A 组后才进入 O_B 等待；AIV 在本组 O_A owner 发布后进行量化，真实 INT8、scale 与所需尾零一并完成后才发布量化阶段。O_B 获取该组全部量化 owner 后可读整 ROW_TILE。组间就绪次序可以变化，组内归约与尾零合同不变。

删除已被上述正文覆盖的“独立候选补充”，迁移状态和实测结果留在测试档案。第 181 行函数 docstring 可明确为：

> 原独立列块路径的全八组入口，复用 o_projection_a_groups_stage_nz；不包含专用 pair 分工。

### cube_schedule.py

模块第 16–17 行“逐组路径不改变本核 O_A 任务序”应替换为：

> 输出尾段按策略选择整阶段或逐输出组通知。逐组 N128 且 0<T≤96 时使用三 pair、两 single 的五 owner 分工，其余区间沿用 logical%24。每核先完成自己的所有 O_A，再消费就绪量化组做 O_B；这保留无环的生产消费方向，但 pair 会改变原每核任务及组间就绪次序。

第 404–408 行 `#` 注释建议统一替换，避免相邻的新旧描述冲突：

```python
# 每核先完成自身全部 O_A，再进入 O_B 等待，避免量化与未来 O_A 互等。
# N128 且 0<T<=96 使用与 Vector 匹配的五 owner；否则保留原 logical%24。
# 两条路径都只在本核负责的组发布一次，发布前完成该组全部本核写入。
```

第 441 行“两分支复用全部数学 helper”应改为：

```python
# 关闭组流水时保留原整阶段数学与通知，不使用 pair 分工及五 owner 等待。
```

### vector_schedule.py

模块第 29–30 行本身未错，可补充精确规则，避免维护者把新 helper 退回通用 owner 公式：

> 组流水启用时，量化只等本输出组的真实 O_A owner。N128 且 0<T≤96 等三 pair 加两 single 的五核，其它区间按原逻辑列块 owner 等待。量化仍按 (group,32 行块)%48 分工；每组末块尾零在同组发布前完成。HCpost 仍等全部 O_B partial，保留八组反量化左折叠顺序。

第 934–935 行可去掉“首版”这一时态；“按本组全部行的真实 owner 等待，不声称逐行最小依赖”仍准确，无需改变运行逻辑。

### phase_sync.py

新 helper 的局部说明已准确；总览第五节可加入：

> wait_oa_pair_group 专用于固定 24 AIC 的 N128 小 T 分工，显式等待 3g..3g+2 和 (3g+3+s)%24。它与原 wait_row_owners 都在获取全部依赖后统一维护业务缓存；两者不能交换 owner 公式。

第四节“参与核集合改变需新工作区”中的参与核建议限定为**调用 begin 的物理核集合**，避免误解成每个阶段的活动 owner：

> 同一工作区的物理参与核集合、核编号映射、阶段槽数及阶段含义必须保持不变。每个阶段的活动 owner 可随 T 和合法策略变化，由当次消费者公式与当代发布匹配；这不要求清空未使用槽。改变物理参与核或槽布局时才需另建初始化工作区。

该澄清也适用于已有动态空通知 guard，不改变任何缓存或代际协议。

