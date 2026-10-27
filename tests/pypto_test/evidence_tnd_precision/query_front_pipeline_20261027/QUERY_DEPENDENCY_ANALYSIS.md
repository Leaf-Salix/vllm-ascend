# 2026-10-27 Query 岛三臂回退定位

## 结论与观测限制

本轮三个实现都是我方实现：Direct 对照、完整屏障 Query 岛、owner 等待 Query 岛。两种 Query 岛均未通过性能门槛，不合入，也没有继续进行 Native 精度验收。

独立dependency运行直接观察到：将Indexer query和主Q放进同一MIX task后，外部Score必须等待整个task完成；内部owner通知不能单独对外发布结果。保留构建条件性关联的里程碑指向TopK/plan前回退，但不能反向证明历史计时的每轮实际registration。

本报告仅作**保留同源构建与独立dependency的条件性离线关联**。原计时未逐轮保存Worker registration到该构建的直接见证；即使构建/CPU/dep一致，模块端点仍依赖该映射假设，不是精确因果证明。完整event与完整L1跨度不依赖函数名映射，性能失败结论不变。没有用旧nalinaly累计cut解释我方task内部耗时。

## 身份与复现

配置为 8K/B32/S6/T192，第4层真实权重、合成 hidden/history，单层完整 HC-pre 至 HC-post FULL graph。12个 event 窗口，每窗100次回放；L1 单独采集12窗，不启用 dependency generation。dependency 在另一任务中按每臂两窗采集，不同时计时。

离线执行需要Python3.11及`regex`，不需要Torch、vLLM或NPU。完整归档恢复方法见本目录`README.md`；在恢复目录执行`python analyze_query_dependencies.py`。本次用已有pre-commit环境（已含regex）复算，没有改默认venv或NPU环境。

输入均保存在本目录 `collected/`：

- `query-island-v14-run1/`：三臂计时和36个原始 L1 窗口。
- `query-dependency-v1/`：三臂独立 dependency，两窗映射一致。
- `query-v14-timing-jit/`：保留的三份计时构建manifest、编排及metadata，缺少逐轮registration直接见证。
- `query-island-v14-cpu/`：最终三臂 CPU 完整64参数资格。

脚本先核验每臂计时 JIT、dependency 运行和 CPU 的全部 leaf name/func ID/记录的二进制 SHA 一致；再核验编排仅本地 `inlineNN` 变量名允许严格双射重命名，保护字面量、注释、预处理和其余全文。`compiled_meta.json` 字节相等。

计时 JIT 的 `kernel_config.py` 是 Artifact.load 序列化后的写法，与 CPU 原生成文件字节不同。脚本只用 AST 白名单解析，逐项核对所有 kernel、source 相对路径、signature、orchestration 和 runtime 配置；已知字符串枚举转换尾循环也必须匹配。没有执行配置导入，也没有宣称配置或编排二进制字节相同。

身份核验通过后，才按 dependency 的 task token 和 core 角色确定该映射假设下的函数，覆盖全部36个 L1 窗口的所有记录。不能直接把 L1 的 registration 字段当 func ID。原始输入 SHA、逐窗结果、逐函数/逐core跨度和命名依赖边均写入 `QUERY_DEPENDENCY_ANALYSIS.json`。脚本重复运行的 JSON 逐字节一致。

## 性能与里程碑

单位 µs，各项分别取12窗中位数；中位数之差不保证可加。下列跨度包含搬运、排队及同步等待，不是纯计算时间。

| 指标 | Direct | 完整屏障岛 | owner等待岛 |
| --- | ---: | ---: | ---: |
| Event完整回放 | 1597.90 | 1667.92 | 1665.34 |
| L1完整跨度 | 1509.37 | 1574.13 | 1580.45 |
| Entry → TopK发布 | 683.57 | 748.21 | 754.99 |
| Entry → TopK + plan完成 | 733.83 | 796.25 | 806.30 |
| plan → heads完成 | 435.74 | 429.65 | 421.19 |
| heads → 结束 | 339.31 | 349.53 | 350.70 |
| Query开始 | 365.44 | 366.71 | 383.08 |
| Query完成 | 550.36 | 596.05 | 608.04 |
| Query包含跨度 | 184.75 | 229.23 | 227.33 |
| Score开始 | 562.23 | 626.94 | 635.39 |
| Score结束 | 643.32 | 707.98 | 715.29 |
| Score包含跨度 | 80.81 | 81.54 | 80.29 |

Direct 的 Query 指六个原独立函数（主 Q_B/后处理、Indexer Q_B/RoPE、Hadamard/量化）的并集跨度；两个候选是整个 `island_cube/vector` 的并集跨度。这里只比较覆盖相同数学工作的总区间，不把它当六个函数耗时之和。

两种岛的 Score 自身跨度仍约80～82µs，但开始时间推迟约65～73µs。owner 等待岛的 TopK + plan 累计完成时间比 Direct 晚72.47µs，已接近其完整 L1 回退71.08µs；attention 后段在该组记录中更短，抵消了部分前段及收尾回退。不能把72.47µs解读成已精确分离的同步成本。

## 为什么减少屏障没有解决问题

实际 dependency 中：

- Direct 的 `indexer_head_coefficients_2` 等待 `qr_hadamard_quant`，随后 Score 消费独立 Indexer query。主 Q 的最终后处理不是这条输入依赖。
- 两种岛都由**同一个 MIX task**生产 Indexer query、量化 scale 和主 Q。该 task 的输出同时连接 coefficients、Score 和 sparse attention。
- 因此 coefficients/Score 对 Indexer query 的 tensormap 等待仍然是整个岛 task；内部 owner notify 并不改变外部 task 完成边界。

| 本轮实际函数 | 开始 | 结束 | 包含跨度 | 最长单core记录 |
| --- | ---: | ---: | ---: | ---: |
| Direct：Indexer量化 `qr_hadamard_quant` | 472.52 | 501.91 | 29.05 | 28.80 |
| Direct：主Q最终后处理 | 495.58 | 550.36 | 55.28 | 52.26 |
| Direct：coefficients | 541.98 | 552.34 | 10.77 | 2.30 |
| 屏障岛：Cube | 366.72 | 595.97 | 229.13 | 229.00 |
| 屏障岛：Vector | 366.85 | 596.05 | 229.20 | 229.00 |
| 屏障岛：coefficients | 607.16 | 616.17 | 8.53 | 8.33 |
| owner岛：Cube | 383.08 | 603.77 | 222.79 | 222.61 |
| owner岛：Vector | 383.15 | 608.04 | 227.28 | 226.82 |
| owner岛：coefficients | 616.03 | 625.40 | 9.32 | 8.58 |

MIX 内各core记录均约220～229µs，只说明 task 占据这些core直到阶段完成；无法用这些包含跨度区分每段 MMA、搬运和轮询。coefficients 的跨度也包含其较大开核时间差，不能把单core2.30→8.58µs直接归因于数学变慢。

## 当时的下一步选择（结果已更新于主报告）

停止扩大 Query 岛融合，也不继续仅在其内部修改等待方式。若要让 Score 提前消费结果，需要改变消费者接入和发布边界，已不是当前这一个同步因素，必须另做设计。

当前更小的候选是 **HC 输入pack的32次独立复制循环静态展开**：只将 `pl.range(32)` 改为 `pl.unroll(32)`，保持原transpose、FP32→BF16、64次slice/insert及尾部清零顺序，不改归约、量化或屏障。预期只影响 `prepacked_prefix_indexer_vector` 一个 leaf。

选择它的依据有限但具体：Direct 的 HC 前缀 Vector 包含跨度190.21µs，并在本轮约190.70µs完成；它生产后续 Q_A/Indexer 所需的 normed 输入，属于前段依赖祖先。该跨度不能证明其中pack占多少时间，也不能预言展开会变快。先确认实际生成循环/资源/同步，再按原0.5% event、10/12胜、L1降低的性能门槛筛选；失败就停止，赢后才做 Native 精度及反向重复。

本轮独立主 Q_B M128/K256 流水已由主线程完成性能筛选，Event回退约0.246%，0/12胜；即使L1略低也不通过，不扩测精度。没有把它与 Query 岛、HC pack 或此前失败的 QA seed/WKV 方案组合。

HC pack和Q_A并行清零已完成正反两轮，均因完整event收益不稳定而未采用；Q_A追加dependency任务获卡前取消，无设备结果。最终状态以主报告及TASKS.json为准。
