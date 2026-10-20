# 当前数学的动态Score/Attention跨请求流水

## 目标、基线与实质区别

正式源仍13259f92/source_after_score_split，五个128K档位降低30%目标保持。
N128宽面板负例已结束，不叠加。当前候选按既有SPLIT_SCORE_ROWS/G6/N64/
1024资源域及本轮device多wave条件选择，不固定B/S/历史或生成静态调度表。

旧score_attention_interleave_current_20261019修复原KV漏执行后，B16全精确
但811.10→861.06μs回退。旧活跃step没有当前物理plan/MGATHER/local统计、
连续scale、新N64早FIX；旁边完整helper更新并不等于活跃路径使用。本候选
真正从当前函数抽取这些数学体，保留原动态wave、ROOT/TOPK/PLAN及首wave
原KV消费。不能将旧结果与本轮跨轮相减，或把当前CPU通过称作性能收益。

潜在收益来自把Cube主导Score与前波Attention的Vector工作交错。当前带
探针B28两段约330/305μs只说明方向值得研究，累计READY不是免费窗口。
片上状态共存、每步恢复Left、wave控制和等待均可能抵消，门槛仍是重点档
至少净省20μs、另一档无明确退步才扩档；没有预设可兑现的微秒收益。

## 阅读顺序与当前实现

1. prepare_migration.py：旧wave控制与当前源三方迁移，保留merge_preview。
   Cube冲突通过当前前后段和当前完整fallback重建；Score冲突保持当前源，
   新step另行抽取，不以旧body覆盖已采用代码。
2. port_schedule.py：当前early geometry、根数/Top-K与physical plan进入
   range接口。只将整阶段获取改由外层本波实际ROOT/TOPK owner获取覆盖；
   wave内不得等待整层SCORE_VECTOR完成，否则形成环。两角色同用SPLIT开关。
3. port_attention.py：从当前完整AIV loop原文抽一拍，只平移4处query首项；
   mask、页保护、MGATHER、softmax、m/l/oi、BF16/逆RoPE和所有事件保留。
   local mi/li为第五份Tile跨拍状态，[6,32] FP32、三槽；无效块不读未写值。
   AIC仍用已验证PV先行/两L1槽形式，GM环三槽，二者生命周期须重新联合验证。
4. port_score.py：当前N64双48KiB QK Acc早FIX、独立4KiB WS和六行一次
   STORE进入Cube step。Query/Coeff驻Mat，每步恢复60KiB Left，避免Coeff
   Left跨Attention保留；额外12KiB系数重装是实质成本，不声称免费。
   Vector step沿当前连续scale/逆序页/尾页清零和原sort/merge，携带真实
   leaf valid，不用rounded steps代替有效长度。布局证明每step重读两个
   不可变Scalar，不重复扫描证明，不改变已发布metadata。
5. finalize_ports.py：完整44源、关键10个调用接口及策略接线检查；这只是
   源级结构证据。audit_codegen.py检查完整分配和真实wave中的生成数学。

前段HC/QA/Q_B/压缩投影、输出后段、Native入参/权重/缓存接口和非交错
数学保持当前版本。signal保留原60个phase前缀，每wave新增ROOT/TOPK/PLAN
独立槽；Host只按输入T容量预留ceil(T/8)，实际边界/owner仍来自device。
不改A3的DCCI/DSB发布获取、100tick退避，不用虚假就绪或全核集合替代依赖。

## 过程问题

首次迁移脚本把git merge-file返回2误当执行错误，实际表示两个冲突块。
已正确接受1..127为冲突数，保存初始部分复制目录与预览；当时没有CPU/NPU
执行或正式源改动。此为本轮脚本问题，没有报告为组件bug。
当前迁移/抽取脚本已执行完成，候选只在本目录；尚未采用到正式ops。

## 完整CPU和生成代码证据

cpu_b16_v1、cpu_b28_v1均完整通过58动态轴、130根、单MIX，无样例shape。
CODEGEN_B16_V1.json和CODEGEN_B28_V1.json确认：

- AIC Mat分配总量507904 B（496 KiB），Left/Right各65536 B，Acc131072 B；
  AIV Vec分配总量188416 B，均在原上限内，未放宽任何预算。
- MGATHER实际3处，其中2处在中间/尾部wave，另1处为完整fallback。
- B16当前区间local统计3处、2处在wave；B28关闭后全部删除。
- B28连续scale载入3处、2处在wave；B16当前区间关闭，全部删除。
- 三套Score代码（两wave调用+fallback）都保留双48KiB QK/4KiB WS独立
  地址、每step32次QK/FIX及16次六行STORE；16面板的早FIX依赖审计通过。

生成审计不是硬件流水重叠、协议、精度或完整L1收益证据，不能预记通过。

## 真机计划与停止条件

先B16/B28正式W8A8层4，原多task/当前正式/候选同轮三方轮转五窗。先执行
原完整精确、cache/state/保护、A/B/A、metadata/padding同址图及真实wave/
基础代际，任一失败即停止，不放宽精度或伪造信号。只有满足20μs与另一档
无明确退步才扩其余目标、S1/S5及区间边界。性能不足则结束此组合，保留
全部结果，不能用阶段插桩或跨轮最优替代完整L1。所有仓库禁止push。

## 最终真机结果：完整正确，但两档均回退，停止采用与扩档

B16 task_20261020_074844_4911330778、B28
task_20261020_074844_491555648均completed/exit0。A/B/A、输出/Top-K/score、
全部cache/state和保护、六次metadata及四次padding同址图重放、真实wave/
基础代际全通过。输出max_abs=0；没有放宽精度或修改同步等待。padding
检查范围沿用原壳，不能扩大宣称验证了空rank、所有混合Q或S1/S5。

|128K档位|原多task完整L1 μs|当前正式SPMD μs|wave候选 μs|较正式净省 μs|更快配对窗|
|---|---:|---:|---:|---:|---:|
|B16/S6|878.56|807.88|850.52|-42.64|0/5|
|B28/S6|1290.20|1167.66|1369.50|-201.84|0/5|

B16五窗配对净省-41.76/-32.98/-46.44/-49.96/-40.52μs，B28为
-198.24/-198.92/-202.54/-200.36/-200.00μs。PERFORMANCE.json保留全部三方
各五窗，数值是各自中位数，逐窗差值也全部保留。未达20μs门槛，不采用、
不扩档、不继续扫wave宽度或局部参数。正式源仍13259f92，最终30%目标未完成。

DEVICE_SOURCE_AUDIT.json确认正式37源与当前基线相同，发布/获取/退避代码
未改，实际设备C++与完整CPU审计版去注释空白后相同；没有以文件名存在
替代活跃机制证据。原候选source冻结保留，部分继承模块开头的“尚未编译”
是迁移时历史状态；其最新验证结论以本节及原始result为准。

## 不新增测试的负例分析

analyze_work.py使用已回读真实metadata和生成IR：B16两波，B28四波。
原完整Score每叶一次恢复60KiB Query/Coeff Left；wave每step恢复60KiB。
按本次真实叶/步数，额外Left操作字节模型分别25.3125MiB、45.9375MiB。
生成IR确实将四份恢复操作置于local_round内；这不是实际带宽计数，也
不能直接用字节除带宽推算节省或回退时间。

B16每核整段Score最大22步，各wave最大步数之和仍22；B28整段最大41步，
各wave最大步数之和56。wave筛选没有改全局owner，但切开范围之后局部
工作分布不再等同于完整范围。这只能提示反复推进波次会暴露长尾；实际
仍按必要前驱等待，并非每wave全核集合，56不能冒充实测关键路径步数。

此外，wave启动/排空、同核MTE/计算竞争和Attention两L1槽与正式三槽的
差别均是实质成本，未单独测量各项贡献。不能把-201.84μs全部归给某项，
也不能从本负例推出一切跨阶段重叠均无效。停止本组合，回到当前正式版
寻找能减少实际计算/搬运的关键路径改动；不再凭两个阶段wall-time较长
就假设有同等长度的空闲引擎可重叠。没有新增设备探针或组件修改。
