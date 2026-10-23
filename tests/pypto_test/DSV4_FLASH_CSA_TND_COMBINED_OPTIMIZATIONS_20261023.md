# Native顺序TND组合优化验收（2026-10-23）

## 目标与状态

用户要求组合“输出融合”和“提前输入打包”，直接测组合是否优于各单项，收益稳定且精度通过后合入同一分支。基准代码6eabea0ec及83c2ad306验收记录保留。当前正在验收，不能把阶段结果写成整模型已通过。

## 正式实现

新包为`deepseek_v4_flash_csa_tnd_precision_spmd_combined`，显式入口`PTO_CSA_VARIANT=tnd_precision_combined`。旧overlap包保持原字节，默认alias保持原选择。中央配置对新alias默认固定规约，显式atomic=1仍由runtime拒绝。

- 输入端：AIV input packing提前到H1/H2之间，与AIC平方和重叠；所有数学、Native K顺序及10次handoff保留。
- 输出端：完整8192列动态量化及INT32 WO-B不变，将末尾反量化的BF16发布与HC-post融合，删除attended中间GM写读和任务交接。
- 四种权重NZ布局及完整64Tensor schema/alias相同，仍覆盖HC-pre到HC-post，算子全部DSL。
- 正式CaptureAttention在捕获前按Tensor总行数选择一个完整PTO根：T≤16旧根，T>16组合根；不在DSL增加分支，不做每请求S6 padding，不回退Native。范围1≤T≤256。
- 新service独立继承已验收的不可变权重/共享workspace生命周期；整模型观察器只包裹正式router记录实际选择并调用原router，不注册或换用实验kernel。

实验复用3399文件固定控制快照、PyPTO2dfa7f120/Simpler57e499238/PTOAS0.66、CANN9.2、Torch2.10.0/Torch-NPU2.10.0.post2/vLLM0.25.1。新正式包、alias和中央配置以逐字节相同的只读import overlay接入，旧SDK与验收快照不改；每文件SHA及计算AST等价均封存。

## 测试方案

1. 无设备编译完整组合根，真实ABI64、外部x_out返回、四权重layout及T1..256路由/schema检查；16项纯CPU测试覆盖短T边界、错T、单入口调用、alias与规约默认。
2. 同卡串行比较Frozen、仅Fused、仅Early、Combined；Native作为相同输入精度参照。第4层真实权重、合成hidden/history，NZ2/page32，FULL graph。
3. 每臂12个event窗口×100次raw replay；四臂轮转，每个位置各出现3次；另采8个L1窗口，每个位置各2次，关闭dep-gen。保留所有样本。
4. 预设组合对每一个对照：event中位数至少降低0.5%、至少10/12成对领先，L1中位数更低。另做反向PTO捕获控制；不相加单项收益。
5. 性能落盘后才验Native A/B八个外部边界及ABA；短T/混长为已胜出组合的范围回归，使用显式qualification，计时后无条件验精度，首差停。
6. 正式接入下复用Native黄金，先8K/B40再128K/B24，各warm192＋测192、TP1/DP=EP16、DSpark出5验6、确定性level1/HCCL开启、EPLB关闭。零抢占、相同接受事件、实际FULL轨迹与21层CSA捕获资格均须通过；HCA/FFN原生。

模型/单层任务内部750秒、调度780秒，不跑Native2，每5分钟内观察。128K采用已验[56,96,176,1]MiB heap/window2048、24000KV块及原2GiB预留；8K保留原ring和22000块。不同ring配置不跨配置计算加速。

## 单层阶段结果

单位μs，越低越好；降低采用`1 - Combined / 对照`。

| 配置 | Frozen | Fused | Early | Combined | 对Frozen降低 | 对最快单项降低 | Native精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8K/B32/T192，原捕获顺序 | 1888.41 | 1841.71 | 1844.78 | 1817.92 | 3.733% | 1.292% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/B32/T192，反向捕获 | 1874.21 | 1831.20 | 1838.06 | 1808.34 | 3.515% | 1.249% | 四PTO臂A/B八边界及ABA逐位通过 |
| 128K/B16/T96，小ring | 1549.78 | 1520.36 | 1533.91 | 1503.71 | 2.973% | 1.095% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/B40/T240 | 2151.10 | 2101.69 | 2095.24 | 2069.71 | 3.784% | 1.218% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/混长1～6/T21 | 779.00 | 770.14 | 794.50 | 774.30 | 0.603% | -0.540% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/全S1/T4，原根保护 | 636.62 | 657.69 | 657.51 | 641.03 | -0.692% | 2.507% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/混长T16，原根保护 | 695.00 | 693.31 | 705.24 | 691.46 | 0.511% | 0.268% | 四PTO臂A/B八边界及ABA逐位通过 |
| 8K/混长T17，组合根 | 758.46 | 748.45 | 761.03 | 762.03 | -0.471% | -1.815% | 四PTO臂A/B八边界及ABA逐位通过 |

首轮组合对三个对照均12/12成对领先；L1中位数分别1778.06/1738.46/1753.22/1713.34μs。反向捕获同样对三个对照12/12领先；两种捕获顺序均有额外收益，不将幅度变化全部归因于捕获位置。128K/B16和8K/B40也对三个对照12/12领先且L1更低，四组全部达到预设组合收益门槛。混长T21没有胜过单独融合版，不能声称所有形状都有组合收益。短T4实际选择原完整根，计时差异不代表计算体改变；该配置用于路由和精度回归。T16实际原根，T17实际组合根；T17比冻结慢0.471%、比融合单项慢1.815%，L1也无收益，完整保留。T16 event差异未获L1支持，亦不算性能成功。路由边界是实现条件，不是整个T>16范围均有收益的声明。整模型尚未全部完成，以后续原始结果为准。

## 启动诊断记录

两次早期硬件启动在Python前exit2并立即释放设备；队列实际会追加`--device 0`。启动器现在仅兼容与TASK_DEVICE完全一致的这对参数，并继续拒绝未知flag、错device和批量/query数不一致；probe也在set_device前再次检查expected batch。七项不含环境/框架/设备的shell header控制通过。初期SSH主动断连和一次自动审批超时保留为执行问题，不算精度或性能结果。

一次`task-submit --timeout 55 --wait`等待超时会自动取消pending任务，原8K任务因此取消且未获得设备；已正常重新提交同配置，之后只用`--status`/`--log`观察。该事件不算硬件/数值失败。

本轮增量pre-commit（含manual Markdown）全部通过。按要求执行的全仓`bash format.sh ci`仍因既有Ruff/拼写/clang-format/历史归档imports等失败；其355处无关自动改动已恢复，本轮受测源码及已stage文件字节完全未变，没有放宽或关闭检查。

本机完整项目UT被原conftest的vLLM依赖阻断；相同正式文件的纯CPU路由测试已在227无设备任务通过，没有改动原conftest或安装本机vLLM。

## 复现资产与参数

| 资产 | 本轮身份/参数 |
| --- | --- |
| 分支 | `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，新优化父提交83c2ad306，冻结数值6eabea0ec |
| 控制源码 | `b28-precision-20261021-codex/whole-b28-qa168-v2/snapshot/source`，3399文件，manifest SHA `48ee0351ebd554c2d52c93d28ff7aa4be357cf6636a4325d247f64a1f2671539` |
| SDK | `/data/pyptouser/yejia/csa-spmd-sdk-20261020`；PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`，Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`，PTOAS0.66 |
| 模型 | `/data/model/DeepSeek-V4-Flash-0731-w8a8` |
| 官方缓存 | 13个静态包，metadata SHA `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602` |
| 8K bank | `low-acceptance-20261010/banks/fixed8192`，plan SHA `13dfb286800d05a8e9105539cd48fd1142d2298e78f4c1693cf8bf2ab5e088e4` |
| 8K Native黄金 | `/data/pyptouser/yejia/lc128/full-pto8k-native-20261016` |
| 128K bank | `low-acceptance-128k-b24-20261014/banks/fixed131072`；plan及每请求token/manifest身份封存在configuration |
| 128K Native黄金 | `/data/pyptouser/yejia/lc128/retest192-native1-b24000` |
| 运行环境 | Python3.12.14，Torch2.10.0+cpu/Torch-NPU2.10.0.post2，vLLM0.25.1，CANN9.2.0-beta.2；具体导入、二进制SHA见CPU及configuration |

相对路径均位于`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex`。新实验在`combined-opt-20261023-codex`。只对bank header/形状/stat和token/manifest身份核验，不声称重新哈希了整份cache payload。复用Native黄金前，五个Native模型/DSA/W8A8源文件逐字节相同。

原始结果、实际脚本、每文件SHA及命令见[证据目录](evidence_tnd_precision/combined_optimizations_20261023/README.md)。实际代码已放入正式package；测试overlay仅将该package及variant/envs映射到相同字节源码，未重装vLLM/SDK。

## 范围与限制

- 本轮单层为真实第4层权重、合成hidden/history，不等于整模型吞吐或token验证。
- 整模型保持TP1、DP=EP16，确定性开启、EPLB关闭，复用固定bank与既有Native输出。不是完整P/D在线服务、EPLB或接受步长3.8达标证明。
- 128K提交B24允许等候入场，目标窗口为21请求、T126、物理bucket144；8K目标40请求、T240、bucket240，实际结果须待本轮完成后确认。全局graph桶padding不等于每请求补S6，短/混query由单层明确验证。
- 小ring只对128K容量资格使用。保护仍要求24000块和2GiB剩余，不降低KV容量门禁。
- 没有本轮新的Native/冻结整模型计时臂，不能据独立整模型计时声称成对加速；单层收益才是本轮受控性能证据。
- 两项优化不覆盖nalinaly与我方的全部性能差距，也不声称追平其SPMD数据。
