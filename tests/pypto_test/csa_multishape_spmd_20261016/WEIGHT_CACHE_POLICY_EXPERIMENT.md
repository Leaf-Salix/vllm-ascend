# 2026-10-17：恢复融合时丢失的权重缓存声明

目标是检查融合单 SPMD 与原动态 PTO 的权重缓存策略是否一致。
不修改数学、调度、精度门槛或依赖仓，不运行 Native forward。

## 发现与边界

原动态 `10_after_OutlineIncoreScopes.py` 对 QA、主 Q、KV、O-A 的只读权重
声明 `cache_policy=[(参数编号, 1)]`；1 对应 `CachePolicy.BYPASS`。
原动态实际 `ptoas/*.pto` 的对应权重 TLOAD 有 `cache_policy` 属性。

实验生成器 `generate_static_entry.py` 合并数学体时过滤原函数的 `pl.func_attr`。
QA、主 Q、O-A 使用 Tensor 数学体重新 lowering，因此这些声明丢失。
KV 使用已经转换的 Tile 数学体，原有两个 `pl.tile.load(..., cache=1)` 仍保留；
所以不能说全部权重的策略都丢了。现选融合实际 IR 共两个缓存属性，位于 KV 读取。
这是实验生成器的行为，当前证据不指向 PyPTO/PTOAS/Simpler 缺陷。

检查 O-A 物理流水时还确认：它已有 K128 内层搬运，Left/Right 在地址 0 与 32768
之间双缓冲，无需再做同样的 L0 双缓冲改写。片段保存为
`oa_selected_physical_fragment.pto.txt`，不把“生成了双缓冲”等同于实际无停顿。

## 候选实现

`weight_cache_policy.py` 从冻结逻辑 IR 的显式声明和本次调用绑定提取四份权重，
检查整个调用图同根视图均未标为 Out/InOut，再按融合后 AIC 参数编号恢复属性。
现有显式 Tile cache 参数保持；没有给 KV cache、state 或临时 Tensor 添加 BYPASS。

BYPASS 的前提是执行期间这些字节没有写者；模型权重满足此合同。
这不是跨核业务数据的发布/等待机制，不能代替原有 fence、缓存维护和进度协议。

开关 `--restore-weight-cache-policy` 默认为关闭，准备合同记录全部原声明及新参数编号。
`weight_cache_policy_prepare_v1` 已通过准备检查；移除新增函数属性后，完整 AST
与现选 `b16_cached_row_deps_v1` 相同。源数学、等待表和 GM 布局保持。

CPU/PTOAS 编译在 `weight_cache_policy_cpu_v1`，完成后必须核对实际 TLOAD/C++
是否含目标缓存属性，再提交正式权重的完整 exact/状态/图检查和三窗无插桩 L1。
当前未有 NPU 结果，不声称性能收益。

## 终态：CPU通过，但生成C++相同，因此未上卡

`weight_cache_policy_cpu_v1`完整CPU/PTOAS通过，实际MLIR的带缓存属性TLOAD由2变44。
然而其`ptoas/csa_static_group.cpp`与现选完整2784561字节文本相同，没有使用hash。
原动态QA的生成代码同样只有普通TLOAD，不能说融合后才丢失有效的硬件BYPASS。

本地PTOAS源码`lib/PTO/Transforms/PTOToEmitC/LoadStore/TLoad.cpp:77`明确要求
`L2Bypass && op.getOffset()`才调整源地址；没有offset时仍生成普通TLOAD。
其`test/lit/pto/tload_cache_policy.pto`也检查这种行为。这与当前生成产物一致，
但不把本地源码HEAD等同于已安装0.66的源码版本。PyPTO的API文档承诺与实际
可见效果有差距；尚未确定应由哪个层补齐运行时offset提供，不将其归为ISA硬件不支持。

A2/A3 PTO-ISA另有`TLoadL2Hint::NotAlloc*`以及运行时填充的
`g_opL2CacheHintCfg.l2Cacheoffset`。当前Simpler动态可调用函数的加载路径与标准
CANN kernel注册不同，未验证该符号能被正确填充；没有硬编码设备别名偏移，也未
修改依赖仓去尝试。此方向保持CPU记录，开关默认关闭，不重复运行等价设备代码。
证据`weight_cache_policy_codegen_audit_v1.json`。初始“全部权重策略丢失”的判断已纠正：
KV显式cache仍在，且标记存在不代表硬件策略生效。
