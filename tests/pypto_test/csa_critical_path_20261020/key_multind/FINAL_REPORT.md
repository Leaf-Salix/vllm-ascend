# Key 多 ND 本轮结论：功能通过，完整性能回退，不采用

2026-10-20。本轮已结束，按用户“这一轮优化之后就先停下”的要求，在本地
归档后暂停总目标。没有开启下一优化方向，没有 push。五档降低 30% 的总目标
仍未完成；本轮负例不能改写成优化成功。

## 完整 CSA 性能

B28/S6/128K，正式 W8A8 第 4 层权重、合成历史。三方在同一进程、同一根
地址、同一新编译器下运行；三方各五窗轮换。统计最早 incore 开始到最晚
incore 结束，保留内部等待，单 SPMD 始终一个 task、72 个硬件参与角色。

任务 `task_20261020_102448_407774415351` 已 completed/exit=0。

| 窗口 | 原多 task / μs | 当前正式 SPMD / μs | 候选 / μs | 正式减候选 / μs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1276.18 | 1162.96 | 1318.58 | -155.62 |
| 2 | 1255.26 | 1162.42 | 1312.18 | -149.76 |
| 3 | 1300.86 | 1163.64 | 1306.16 | -142.52 |
| 4 | 1292.24 | 1171.06 | 1310.50 | -139.44 |
| 5 | 1289.56 | 1166.86 | 1307.86 | -141.00 |
| 各方中位数 | **1289.56** | **1163.64** | **1310.50** | **-146.86** |

0/5 配对窗更快，候选比正式版慢 12.62%；相对多 task 的耗时降低比例为
-1.62%。不满足“重点档至少净省 20 μs”的门槛，因此停止、不采用、不扩
B16/其他档位，不增加参数扫描。1163.64 μs 是本轮正式参考值，不能与历史
1158.92/1175.34 跨轮相减归功于本候选。没有新测 Native、静态 SPMD、8K。

`PERFORMANCE.json` 和 `device_b28_v1/source_matrix/result.json` 保留全部
原始数字；`report_results.py` 只接受任务终态及完整验证通过的窗口。

## 实际修改

- 正式 `ops/pypto/deepseek_v4_flash_csa_spmd` 的 37 文件保持
  `source_after_score_split/13259f92` 内容。本候选完整 37 文件保存在
  `source_candidate`，只改变其中 `indexer_score_stages.py`。
- 两个正序相邻物理页且不重复尾页 clamp 时，一次多 ND 搬入 N64 Key。
  Native 连续根保持不变，incore 内建立 `[P,32,128]`、stride
  `[page_bytes,128,1]` 视图跳过页尾 scale。页宽、物理页和尾页均有 device
  分支保护；不适用时保留原逐页读取。没有新 Host 页表值、Key 拷贝或布局转换。
- 独立 PyPTO 树 `.worktrees/pypto-csa-key-multind-20261020` 基于
  `2dfa7f120ba5e577566046d920845aa241bfa7bb`。14 文件增加显式 stride view、
  A2/A3 多 ND 能力开关、保留三维源窗口的 flatten、UT 和中英文文档。
  `pypto_candidate_v2.patch` 与 `pypto_source_v2` 是最终完整补丁/对应文件，
  v1 保留第一次 UT 修正前的状态。没有改共享安装、原 PyPTO 树、Simpler、
  PTOAS 或 PTO-ISA，没有复制扩展二进制。
- 新接口只支持 incore 内局部、完全有效的 ND 别名，编排侧明确拒绝。
  静态跨度/重叠/溢出检查不替代符号尺寸的运行时调用合同；原根 contiguous
  校验保留，不支持任意 stride 的 Host 根输入。

## 编译和设备证据

1. 本树完整构建成功，并发由 testing loader 限定为 2。UT 首轮 53 通过、
   1 个测试写法失败：三维 Tile 与二维 reshape 复用了同一变量名。改为
   `flat_tile` 后 54 项全部通过，未改预期或放宽约束。
2. CPU v1 在 view 中内联 `tensor.dim`，描述符 shape/stride 出现空字段。
   原配套 PyPTO、没有新 stride 接口的 `inline_dim_repro.py` 也能复现。
   暂改为先绑定 device Scalar；没有改成 Host 数据或样本常量。
3. CPU v2 试图将多 ND Tile assemble 到原 128 行父 Mat 双槽，生成 Mat→Mat
   TMOV，被当前 PTOAS 拒绝。该方式停止，失败源 `probe_pool_v2.py` 保留。
4. CPU v3 用独立 N64 Mat 直接送 Right，生成的 Key TLOAD 确实是
   `[2,32,128]`，保留原页 stride。基础任务
   `task_20261020_101730_37553223693` completed/exit=0：页宽
   4160/4224/8192 × 同图页号 0→1→7→0 共 12 项精确点积、输入不变和输出
   保护通过。这个 probe 没有采性能，不代表完整 CSA 或所有页表边界通过。
5. 完整候选及新编译器下正式版 CPU 编译通过，58 符号轴、130 根参数、
   单 MIX。AIC Mat 475136 B、Left/Right 各 65536 B、Acc 131072 B；
   AIV Vec 163872 B，均未因候选增加峰值。
6. 完整 B28 的三次 A→B→A、冻结输出/Top-K/score/cache/state/保护检查、
   六次同址 metadata 图重放、四次 padding 图重放及性能后代际检查通过。
   未覆盖混合 query 长度、空 rank、S1/S5 或其余档位，不宣称全动态范围通过。
7. `DEVICE_SOURCE_AUDIT.json` 直接核对实际设备 PTOAS 输出和同一 100 tick
   等待变换后的 C++ 与 CPU 审查源相符；没有阶段时戳。正式源保持不变。
   `FULL_CPU_AUDIT_V1.json` 还确认正式版本在新旧编译器下 C++ 去注释/空白
   后一致。均不声称机器码逐字相同，不做 hash 校验。

## 回退解释的证据边界

多 ND 减少了一组 Key load 的发射次数，但最终实现同时改变了 Mat 复用。
原 128 行父 Mat 的显式槽组装不可用，候选改成 N64 Mat 直接进入原两个
Right 预取槽。实际生成代码中这两个 Right 槽仍是 49152/57344、各 8192 B；
prime/循环中的 N64 Mat 是 65536/114688，循环下一块复用同一地址。

这些生成物证明不能把候选描述为“原双缓冲完全不动、只减少一次 DMA”。
Mat 复用可能改变 MTE1/MTE2 依赖；三维描述符和 device 分支也可能有成本，
但本轮没有足够核内时序证据对 146.86 μs 做定量拆分。不得断言退步全部
来自 multi-ND 指令、编译器错误或某条等待。依用户要求不再追加诊断轮次。

## 上游记录、复现和归档

- 旧接口的内联 `tensor.dim` 复现和建议已补充到 `INLINE_DIM_2951_COMMENT.md`
  对应的 #2951 评论；明确没有重新构建干净最新 main。
- 显式 stride 和多 ND 原型、通过范围、完整性能负例及建议补充到
  `UPSTREAM_3038_COMMENT.md` 对应的 #3038 评论。URL 分别保存在同名
  `*_URL.txt`。没有提交修改 PR，没有加标签，没有 push。
- 独立工具链路径见 `WORKTREE.json`、`BUILD_COMMAND.json` 和
  `run_isolated.sh`。复现先按基线准备独立树并应用 `pypto_candidate_v2.patch`，
  再运行 `build.sh` 和 `run_ut.sh`。`probe.py --cpu-only` 可只编译；完整
  候选的控制参数和编译结果见 `controls_b28.json`、
  `cpu_csa_b28_v1/result.json`，完整设备测试由下述 `run_b28.sh` 进入原测试壳。
- 设备通过 `task-submit --device auto` 调用 `run_probe.sh`、`run_b28.sh`；
  重跑必须提供新的 `--output`，不要覆盖已归档目录。本轮不再实际提交这些任务。
- `archive.py` 选择完整算子源、两版组件补丁/源、所有失败日志、结果及
  15 个原始 L1 窗。大 IR/C++/build/二进制仅本地保留并在父目录
  `LOCAL_ARTIFACTS.json` 登记路径、文件数和字节数；不提交权重和二进制。

本轮所有设备任务均终态。暂停后除非用户重新授权继续，不自动开启优化。
