# PV 列 tile 区间策略接线：历史候选，最终未采用

## 最终状态：验证完成，不采用，不再扩测

根代理已完成七个目标档与短历史 B20 区间点。各点完整功能/精度检查通过，
但长历史 G6 域内收益并不一致，短历史域均回退。因此**正式不采用 N256 区间
策略，不制造 B4/B16 白名单，不再增加边界补测**。本目录保留四文件接线、
33 源前后快照及两次 CPU 证据，全部只作为历史候选。

最后三档按 `statistics_us` 的角色键核对如下；下表为更正后的正确方向，
不按 grep 打印顺序猜参考/候选角色：

| 长历史档位 | 参考 N128 / μs | 候选 N256 / μs | 相对参考 | 更快窗口 |
| --- | ---: | ---: | ---: | ---: |
| 128K B8 | 677.70 | 673.82 | 降低 0.573% | 3/5 |
| 128K B24 | 1186.54 | 1183.62 | 降低 0.246% | 3/5 |
| 128K B28 | 1352.18 | 1355.54 | 回退 0.248% | 1/5 |

完整八点、五窗口、任务及策略由主仓
`tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/pv_n256_evidence/SEVEN_AND_INTERVAL_RESULTS.md`
及同名 JSON 记录。其功能/精度通过不等同 Native/token 或全部动态场景验收；
七档均未达到相对原动态分 task 降低 30% 的目标。

以下保留当时接线设计和 CPU 结论，不删除“待采用/待长域补测”等历史文字。
这些文字只描述当时快照，后续采用状态以上述最终结论为准；本次没有更改
冻结源码、runtime、两次 CPU 或原始生成证据。

## 当时的接线设计与 CPU 记录

2026-10-18。`pv_cols` 从 Host 策略经单 SPMD 入口传给 Attention Cube helper，
字段和尾部 constexpr 的默认值均为 128；当前私有选择器仅在
`long_history && group == 6` 时选择 256，其余仍为 128。**这是待完整实测采用的
候选，不是已合入正式的策略，不声称已经证明性能根因。**

两种显式列 tile 各做一次完整符号 CPU 编译，均 PASS：130 个 runtime 根 Tensor、
58 个动态轴、单 MIX、sync_start 保持，不传样例 shape。与已经跑过的旧 N128、
旧 N256 同控制实际产物对比，完整 Tile 分配序列、地址/shape、compiled_meta、
数学/搬运/同步调用顺序均相同。去掉 `// pto:` 调试注释后的完整 C++ 逐字一致。
本轮无设备任务，不把接线编译通过当作新增精度、性能或混合 Q 场景的验证。

## 范围与区间含义

基线来自 canonical 正式包的 33 个 Python 文件，制作时 Git HEAD 为
`8d46159f392a4d621a63dd20e8f84a1fe982e9f5`。本轮 `source_before/` 与既有
`parallel_20261018/pv_n256_h131072_b16_v1/source_matrix/reference_source/`
33 份源码逐字相同。`source_before/` 与 `source_candidate/` 自两轮 CPU 启动前
冻结，冻结时间见 `source_manifest.json`。不修改正式仓、共享安装、依赖或主测试脚本。

原策略的 `long_history` 是 Native 已有 Host 长度上界推导的
`max_seq_len // 4 > 8192`，不是回读设备实际长度。G6 仍由既有工作量逻辑选择；
此长域 G6 同时使用 balanced 单根 Score 编排。PV 列 tile 区间复用这个已有
编排/资源域，不根据七个 B/S 验收点列白名单，也不把 T、B、Q/L 写死。

选择该域的依据只是既有区间对照结果分化，需要继续由根代理的完整长域补测
决定是否采用；不能把 G6、资源安排或任何单项都说成已证明的性能根因。短域
B16/T96、B20/T120 的 N256 结果均不支持启用，不能仅按 T、early-KV 或输出
tile 阈值把短域也打开。本轮不新增或改动性能结果，根代理统一记录原任务。

## 四份源码的最小变化

| 文件 | 变化及边界 |
| --- | --- |
| `policy.py` | 新增 `pv_cols: int = 128` 与 `PV_N_TILE` 映射；检查精确 int、取值 128/256；选择器显式返回上述待采用区间。旧手工构造策略不传新字段仍为 128。 |
| `entry.py` | `decode_csa_spmd` 和 `_csa_core_group` 末尾新增默认 128 的 constexpr；只传给 Cube，不给 AIV 添加无用参数。 |
| `cube_schedule.py` | 尾部同名 constexpr 默认 128，转给 Attention Cube helper；实际 query 循环、同步和 owner 均保持。 |
| `attention_stream_stages.py` | 将文件级 `PV_N_TILE=128` 改为 helper 尾部同名 constexpr 默认 128。原 extract、列 unroll 与 store 偏移共用参数，AIV 原体逐字未改。 |

`binding.py` 已使用 `**self.policy.constexprs()`，无需新增手工接线。
PV 是完整 512 列的独立分块，128/256 都整除 512；K 仍为 128。softmax 的
候选块、三槽 KV/P/score/PV GM 布局、AIV 两段 256 列消费及全部 READY/drain
事件均不变。所有列 store 完成后仍在原位置发布 PV_READY，不依赖执行时间
代替缓存可见性。没有新增运行时标量、Tensor、phase、Host 回读或容量调整。

## 源级与生成物验证

`check_source.py` 去掉新增 constexpr 声明/传参、原常量与 docstring 后，对
三份执行源做完整 AST 比较，全部一致；没有消去任意算术、分支或业务调用。
策略验证覆盖 84 个 Host 描述符组合，包含长/短、S1/S5/S6 和 G2/G6；7 个
非法值/类型被拒绝，默认构造保持 128。这是 Host 策略检查，不是设备覆盖声明。

完整 CPU 使用主脚本已有 `--controls-json`，分别明确给出
`controls_128.json` / `controls_256.json`。其他控制严格匹配旧长 B16 实产物：
G6、QK64、Score1024、balanced/root1、OA128/OB96、行依赖开启、query metadata
和 HC consumer reduce 关闭。没有利用 CPU 默认的 OA256/OB128 冒充同控制对照。

| 检查 | N128 接线对旧 N128 | N256 接线对旧 N256 |
| --- | --- | --- |
| 完整符号/后端 CPU | PASS（一次） | PASS（一次） |
| runtime Tensor 根 / 动态轴 | 130 / 58 | 130 / 58 |
| 单 MIX / sync_start | 保持 | 保持 |
| 完整 compiled_meta | 相同 | 相同 |
| 完整 Tile 物理分配序列、地址、dtype、shape | 相同 | 相同 |
| 数学、搬运、FFTS 与本地同步调用行及实参顺序 | 相同 | 相同 |
| 去掉 `// pto:` 调试注释后的完整 C++ | 逐字相同 | 逐字相同 |

两对的片上地址上界都相同：AIC Mat 475136、Left 65536、Right 65536、Acc
131072、Vec 16384；AIV Vec 163872 字节。这是现成分配器产物的地址上界，
不据理论容量推断性能。完整分配列表按次序比较，不仅比较峰值。

原始 PTO / C++ 并非逐字相同：inline SSA 的调试编号和源码位置随进程/新文件
变化。`generated_audit.json` 如实保留原始不等项；原 C++ 只有 `// pto:` 注释
变化，业务变量、表达式、控制流、指令调用和同步原文全部相同。保存了每种模式
前 240 行原始文本 diff，以说明差异来源，没有将 SSA 名称差异当作数学变化。

旧产物根目录：
`parallel_20261018/pv_n256_h131072_b16_v1/build_output/` 下
N128 为 `decode_csa_spmd_q1m9wnpj`，N256 为 `decode_csa_spmd_zpf_6y0x`。
新产物为本目录 `cpu_n128_v1/kernel/`、`cpu_n256_v1/kernel/`。

过程没有编译失败。生成审计脚本首次执行有一个多余括号导致 SyntaxError，
已仅修正审计脚本并重读现成文本；没有重编候选或更改冻结源码。

## 控制入口与复现

旧手工 controls 不传新项时，入口/helper 默认 128，因此本轮无需修改主 CPU
脚本。显式覆盖名称为 CPU `PV_N_TILE`、device policy `pv_cols`。现有 CPU
字典 update、设备 dataclasses.replace、binding.constexprs 都不会过滤新字段。
source matrix 的 reference 独立加载参考包策略，继续旧 128，不误用候选覆盖。
后续正式整合时仍建议补主 controls/接口清单文档；本轮按授权没有改它们。

以下是已执行的两次 CPU 命令，结果目录禁止覆盖，不应为了复核报告再执行：

```bash
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_pv_policy_wiring_20261018
bash "$D/reproduce_cpu.sh" "$E/cpu_n128_v1" --strategy long \
  --attention-query-metadata off --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls_128.json" > "$E/cpu_n128_v1.log" 2>&1
bash "$D/reproduce_cpu.sh" "$E/cpu_n256_v1" --strategy long \
  --attention-query-metadata off --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls_256.json" > "$E/cpu_n256_v1.log" 2>&1
```

仅读取证据可以重跑 `python check_source.py` 与
`python audit_existing_codegen.py`，不启动编译或 NPU。完整核源码和日志本地
保留；二进制不提交，大小/路径见 `LOCAL_ARTIFACTS.json`。未生成 hash，未提交或 push。
