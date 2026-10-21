# CSA-SPMD已提交issue：官方修复接入记录

按用户要求，从CSA-SPMD session及其三会话issue清单核对官方修复。
GitHub状态以本目录`remote_status.json`的实时API查询为准，浏览器缓存不作为合入证据。
来源清单：`tests/pypto_test/csa_three_session_issue_status_20261021/STATUS.json`。

当前状态：**源码已整合，三个session的公共入口已切换；验证仍在进行，PTOAS尚在构建**。
PyPTO及torch_npu桥接构建通过，核心回归891项、代码生成152项通过；当前尚未完成新PTOAS与设备验收。
新环境继续以kernel调试分支为基线，针对已合入main的修复按依赖移植。
用户明确要求三个session统一使用本地版本，并已停止另外两个session以便切换；
公共选择改由`tests/pypto_test/csa_shared_toolchain/manifest.json`提供，不再各选私有SDK。

| PyPTO issue | 官方修复PR | 本地提交 | 内容 |
| --- | --- | --- | --- |
| [#2951](https://github.com/hw-native-sys/pypto/issues/2951) | [#3004](https://github.com/hw-native-sys/pypto/pull/3004) / `a1702ad4` | `f803412` | tensor.dim保留形状表达式 |
| [#3009](https://github.com/hw-native-sys/pypto/issues/3009) | [#3013](https://github.com/hw-native-sys/pypto/pull/3013) / `2964333d` | `57384ea` | while条件循环状态存活 |
| [#3014](https://github.com/hw-native-sys/pypto/issues/3014) | [#3018](https://github.com/hw-native-sys/pypto/pull/3018) / `acc125ce` | `9dd54b1` | Group返回值与共享参数绑定 |
| [#3015](https://github.com/hw-native-sys/pypto/issues/3015) | [#3019](https://github.com/hw-native-sys/pypto/pull/3019) / `905ee5e2` | `6611d0c` | phi后的reshape/transpose_view来源 |
| [#3016](https://github.com/hw-native-sys/pypto/issues/3016) | [#3029](https://github.com/hw-native-sys/pypto/pull/3029) / `ec064a82` | `9026aa9` | 条件分支共享Tile窗口及作用域 |
| [#3026](https://github.com/hw-native-sys/pypto/issues/3026) | [#3030](https://github.com/hw-native-sys/pypto/pull/3030) / `57821c37` | `dbf51c1` | 有符号整除/取模全链路一致 |
| [#3027](https://github.com/hw-native-sys/pypto/issues/3027) | [#3031](https://github.com/hw-native-sys/pypto/pull/3031) / `69d5afb8` | `c50456c` | 存储重绑定后语义别名维护 |
| [#3017](https://github.com/hw-native-sys/pypto/issues/3017) | [#3053](https://github.com/hw-native-sys/pypto/pull/3053) / `cb7e1778` | `a33f14c` | 共享目的buffer的物理类型视图 |
| [#3041](https://github.com/hw-native-sys/pypto/issues/3041) | [#3092](https://github.com/hw-native-sys/pypto/pull/3092) / `fb939f5d` | `2b05e98` | 内联自动名称role/version保持 |

其中8项是CSA-SPMD新提交后关闭的bug；#2951是其跟进的既有issue。
PTOAS直接使用master `2a2bad22`，已含 [#1620](https://github.com/hw-native-sys/PTOAS/issues/1620)
动态地址同步修复与 [#1622](https://github.com/hw-native-sys/PTOAS/issues/1622) CFG并行赋值修复。
#1622的官方代码已进入master，即使对应PR #1642仍显示open，也按实际修复提交核实。
PyPTO #2886属于说明后关闭，没有新增除法指令实现，不能声称高精度除法能力已修复。

迁移时保留CSA依赖的native 150tick、动态Tensor签名、CORE256 ABI、精确子视图同步、
A3标量比较Gather和嵌套位运算语义。与官方修复重叠的旧补丁采用官方完整规则；
每项适配及未引入的父分支无关功能记录在详细中文commit中。

- PyPTO：`.worktrees/pypto-csa-mpmd-upstream-20261021`。
- PTOAS：`.worktrees/ptoas-csa-mpmd-upstream-20261021/build`。
- Simpler：继续`.worktrees/simpler-csa-core-args-256-20261016`的`57e49923`。

主仓及需要网络拉取的子模块均浅克隆；Simpler源头从本地固定提交浅克隆，只用于编译头文件。
PTOAS使用现有LLVM19构建；CANN CMake依赖只复用已固定的master-055源码，不拷贝旧工具产物。
新PyPTO/PTOAS在各自目录构建，使用testing工作流默认的2个构建/测试进程。
PyPTO使用RelWithDebInfo的g1（保留栈回溯），PTOAS Release，不开启Python binding。

待完成：定向官方回归、冻结CSA的CPU编译、单卡SPMD/MPMD数值与图重放及同轮L1。
公共入口在PTOAS产物缺失时拒绝启动，避免悄悄回退旧版本。升级后的数据不能与旧版本拼成性能收益。
完整版本与状态见[manifest.json](manifest.json)。
