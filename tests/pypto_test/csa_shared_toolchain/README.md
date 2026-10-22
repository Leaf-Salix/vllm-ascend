# 三个CSA session共用的本地工具链

新环境从源码进行 PyTorch 单卡复现请使用
[2026-10-21 复现说明](../DSV4_CSA_REPRODUCTION_20261021.md)。
其中固定 Git SHA，并提供 PTO 工具链构建、独立单卡检查和 v25 实验包的重建命令；
无需安装 vLLM、vLLM-Ascend 或编译 Native 自定义算子。

CSA-SPMD、CSA-SPMD-TND、CSA-MPMD-TND共用本目录的`manifest.json`。
当前选择`csa-shared-20261022-logical-codegen`，状态为`validated`：已完成独立构建、
113项编排回归和大容量CSA CPU编译，128K/B24/S6及混长B16的单卡数值、状态和图验证通过。PyPTO仅增加官方#3000的
逻辑表达式生成修复；PTOAS和Simpler不变。详见
[升级证据](../csa_mpmd_aicpu_20261022/upstream_logical_refresh/manifest.json)。
大容量MPMD的AICPU逐页校验仍存在性能问题，SDK验证通过不代表算子性能达标。
此前`csa-shared-20261021-upstream-fixes`的源码和构建产物原样保留，已启动进程继续
使用启动时绑定的路径；新启动器统一读取当前清单。历史结果仍标记原版本。
其中路径同时指定源码和实际构建产物；目录名保留开发时名称，三个session不再各选一套SDK。

```bash
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
source /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh
python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
  --shared-toolchain /path/to/test.py
```

公共入口已接入CSA-SPMD的`csa_roofline_optimization_20261020/run_isolated.sh`、
TND SPMD的`csa_spmd_tnd_20261020/run_probe.sh`/`run_cpu.sh`，以及
TND MPMD的`csa_mpmd_tnd_20261021/launch.sh`。
工作进程继承统一的Python模块查找路径，优先于历史editable安装。
入口打印实际选择；BSH旧清单只保留算子源码参考，新的SDK选择由公共清单决定。

独立 worktree 通过向上寻找 `env-dsv4-0251rc1.sh` 定位工作区，仍读取主检出
`vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json`。
TND 的 `run_compile_cpu.sh` 和 `run_isolated_probe.sh` 也使用该公共选择；
历史 `runs/` 中冻结的启动器及结果保持原版本。

`status=validating`表示本轮升级仍在验证，不能作为已验收版本使用。
若PTOAS产物缺失，公共入口立即报错，不从PATH退回其他版本。
完成定向回归和受影响单卡运行后，将版本及证据记入清单。
新性能对照的各方必须在同一进程/任务使用本清单；历史结果仍保留当时版本，禁止跨版本算收益。

SDK源码修改、重建和下一次升级由公共清单统一发布，不在有排队/运行任务时原地编辑已发布SDK。
算子实验仍可使用冻结的私有包；历史冻结启动器和结果不改写。
不进行反复逐文件hash扫描，来源检查以版本、路径和实际加载产物为准。

本轮修复明细与验证状态：
[`upstream_refresh_20261021`](../csa_mpmd_tnd_20261021/upstream_refresh_20261021/README.md)。

一次性实际加载来源检查可通过公共bootstrap运行`show_versions.py`；它同时核对父进程和子进程。
`compile_entry.py`只做CPU编译；`run_ptoas_regressions.py`验证本轮采用的官方修复与配套补丁。
这些辅助检查不在每次性能测试前重复执行。
