# Q_B 全局 N128 最小候选（完整验证通过，性能无收益不采用）

候选与正式 `66882634` 的差异严格限于两个文件：NZ 的列块宽度从 256 改为 128，Vector 的 QB 等待改用派生常量 `QPROJ_N_BLOCKS`。33 个 reference 源已从该提交冻结，并与当时正式工作区逐文件直接比较一致；candidate 的其余 31 文件字节一致。准备阶段仅执行源码复制、AST审计和纯整数模型。主会话归档后已执行一次完整CPU，result.json为PASS；尚未提交设备任务。

`source_reference/` 和 `source_candidate/` 已冻结。没有混入上游单 owner 等待、Attention geometry、OA、QR/B 驻留或其它候选。

## 调整内容和不变量

- `query_projection_stages.py`：`QPROJ_MM_N_TILE = 128 if QUANT_WEIGHT_NZ else 512`；更新中文总览和 head 列块说明。ND512 保留，NZ 仍是动态 M128、四段 K256、原 stage2 和 INT32 累加顺序。所有函数 AST 完全一致，只有模块常量与总览文案改变。
- `vector_schedule.py`：导入 `QPROJ_N_BLOCKS`，替换 `PHASE_QB` 等待中的旧 count128；补充 NZ/ND、全部 owner 获取的中文说明。发布、fence/cache invalidation、DQ 数学和任务分组均未改。

NZ 的全部 32768 列现在是 256 个 N128 任务，owner 仍为 `n % 24`，每个 owner 内遍历该 N 块的全部动态 M128 行块。这不是在原 N256 owner 内部计算两条 N128 链。每个 head 的四个 N128 列块可能由不同核写回，后继仍先获取全 24 个 AIC 的 QB 完成通知。ND 的 64 个 N512 列块也覆盖全 24 核；`wait_row_owners` 对任务数去重后，真实等待集合不变。

## 模型和源码检查结果

`audit_source_model.py` 直接解析候选的 N 轮数、N 索引、M 上界和 `m_rows` 表达式，以标准库解释整数表达式，不执行候选模块。reference/candidate 各枚举 T=0…4096，检查全部列块恰好一次、列块 owner、动态 M 尾有效行连续且无重叠/遗漏。T0 只是整数边界，不宣称现有入口支持空输入。

`SOURCE_MODEL_AUDIT.json` 记录 PASS：

- 两侧各 4097 个 T，24 worker；有效 `(token, column)` 写覆盖完整且唯一。
- NZ 和 ND 获取集合均为 0…23，列写回起点按 64B 缓存线对齐。
- 精确反向文本替换恢复原 33 源；独立逆变换 AST 恢复原 33 个完整模块。
- 全部 Query 函数 AST 不变；逆变换只恢复 NZ 常量、QB wait 参数和新增 import，总览文案由精确文本替换约束。

| 工作量 | 正式 NZ N256 | 候选全局 NZ N128 |
| --- | --- | --- |
| 每核列任务数 | 核0…7各6，核8…23各5 | 核0…15各11，核16…23各10 |
| 最大/最小输出列数 | 1536 / 1280 | 1408 / 1280 |
| 一 M 波每核 MMA（重/轻） | 24 / 20 | 44 / 40 |
| 两 M 波每核 MMA（重/轻） | 48 / 40 | 88 / 80 |
| 全局 MMA / FIX 次数，M=ceil(T/128) | 512M / 128M | 1024M / 256M |

最大每核有效点积量降低 8.33%，但核8…15从1280增至1408列。总点积、输出字节和权重逻辑载荷 `32MiB × M` 不变；QR 逻辑载荷翻倍，循环/MMA/FIX 数量增加。T96 的 QR 12→24MiB、T192 的 QR 24→48MiB；这些是源码请求量，不是实测 HBM 流量，更不是时延收益。

## 历史不是空白；双槽必须看真实产物

旧静态 20worker/T96 已经做过全局 N128：主仓 `tests/pypto_test/csa_multishape_spmd_20261016/qproj_n128.py:25–56` 把列任务128→256、列偏移256→128并重新分配 owner。对应 `HC_MIX_QPROJ_EXPERIMENTS.md:37` 和 `qproj_n128_codegen_audit.json` 记录完整 CPU PASS，但四个 Right 均为地址0，`two_right_buffers_present=false`，未上设备。因此不能称全局 N128 是从未试过的思路。

既有 `dynamic_full_pipeline_20261017/parallel_20261018/qb_n128_l0_readonly/README.md:41` 和 `QPROJ_L1_COLUMNS_EXPERIMENT.md` 是原 owner 内双列方案，不是当前24核动态全局重分的设备失败。本轮排重结论及精确路径还见归档的 `history_review/README.md`。

理论单块资源从 Right64→32KiB、Acc128→64KiB；Left32KiB 不变，两 Right 容量上可以合计64KiB。不能据此预记真实双槽、重叠或收益。一次完整 CPU 后应核对动态 valid 尾、真实 NZ 切片、Left/Right/Acc/Mat 地址及生命期、覆盖前等待、首 K 与后三 K 的顺序，以及非 QB 数学/同步。若 Right 仍同址，如实记录，不自动追加手工预取或参数变体。

## 归档与复算

复制本目录即可：`source_reference/`、`source_candidate/` 各33个 `.py`，以及 `prepare_source.py`、`audit_source_model.py`、`candidate.diff`、`SOURCE_PREPARATION.json`、`SOURCE_MODEL_AUDIT.json`、本 README。`SOURCE_PREPARATION.json` 保存每个源文件的路径及字节大小，不使用摘要校验。

`prepare_source.py` 是构建记录，已存在冻结目录时拒绝覆盖；可在另一个全新证据目录复现。`python -B audit_source_model.py` 可在当前目录复算纯源码/整数检查。CPU 和设备入口由主会话管理，本目录没有另建包装。


## 主会话进度

唯一CPU命令为 `bash run_cpu.sh`，结果在 `cpu_v1/result.json`，日志在 `cpu_v1.log`。symbolic parse/full codegen/single MIX/sync_start均PASS，保留130 Tensor根、58动态轴，未提供静态sample shape。生成内存和事件仍在独立审阅，不能从CPU通过推断设备精度或性能；不重复编译。完整编译缓存与生成物见CPU_ARTIFACTS.json本地路径清单。

独立生成审阅已完成，详见 `generated_review/README.md`：真实Right仍同址；非QB归一PTO仅8条Acc地址变化，AIV全文不变。新增3处非QB MTE1 barrier须作为后续完整L1归因限制。已提交长128K/B16、短8K/B32两代表，任务与参数在registry.json；待完整设备验证，不预记收益。

## 设备收口：不采用

两代表完整功能/精确/图/状态/保护PASS，但正式668→候选分别881.40→882.74μs、1004.20→1005.74μs，均无改善。详见DEVICE_RESULTS.md/json、ADOPTION.json。正式源不改，不扩测七档，不追加双Right变体。
