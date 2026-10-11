# 过程084：嵌套分支手工 Wait 归并

2026-10-12，单卡0、CANN 9.2、TMR。只扩展手工边界可表达的控制流，不自动寻找 Wait 位置。
私有 PyPTO `1085814eb` 保留本次修改；Simpler/PTOAS/PTO-ISA、原派发/ACK/early-resolve、
chip swimlane/PMU/参数 dump 的串行回退均未改变。公共 CSA 默认未切换。

## 修改与边界

原实现只会拆分顶层双分支中的直接 Wait。嵌套分支在 Wait 后还有操作时，即使所有路径已标明
边界，也无法归并到顶层。本次先索引源 Wait，递归拆分已标注的分支，再在最外层统一替换引用：
只执行选中分支的 Scalar 前缀，条件只求值一次，保留原后半段及其 SSA 输出。
同类型 Scalar 复用传递槽；编译器生成的参数加载和 MIX 初始化仍在公共 Wait 后。

不允许缺失/重复边界、循环内 Wait、Tensor/tile 捕获，也不允许把输入 GM 读取放进前缀。
索引与源遍历为线性；构造还取决于生成的 SSA 传递边数 K，整体 O(N+K)，
不按嵌套深度反复扫描/替换已有子树。很深的嵌套及大量活跃 Scalar 仍可能扩大生成 IR，
本轮没有编译规模压力测试，不宣称任意嵌套代码都能低成本转换。

[完整补丁](pypto.patch) 包含生产转换、结构对照、设备夹具和中英文接口说明；共6文件 +442/−166行，
其中生产转换 +167/−78行，其余为测试和文档。MIX 测试工厂保留原 `mixed_chain_probe` 入口供历史诊断调用。
[版本](versions.json)；[结果摘要](summary.json)。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 新结构对照在修改前 | 失败：嵌套边界仍留在原分支内 |
| ConvertTensorToTileOps 回归 | 300项通过，约5秒 |
| 受影响正/负向编译检查 | 7项通过；含非法 GM 读取、缺失/重复 Wait |
| 单卡纯 AIC，两次输入 | 每次24/24个同核尾部交接时序通过 |
| 单卡纯 AIV，两次输入 | 每次48/48个同核尾部交接时序通过 |
| 单卡 MIX，两次输入 | 每次24 AIC、48 AIV在前驱组完成前进入前缀 |
| 输出、分支 Scalar、GM 观察值、保护区 | 全部通过 |

AIC/AIV 的观察包含运行时完整尾部等待及通知，不能单独证明原始 Cube/Vector 计算仍在运行；
MIX 数据是前缀入口早于前驱组完成的覆盖证据，也不等于每个成员都发生计算重叠。
测试改变输入再运行两次，没有新增性能、Native、token、16卡或七档结果。
现有 CSA 的73个成员均为顶层 Wait，不触发本次分支改写，沿用过程083性能数据。

[修改前失败](before.log)、[结构对照通过](after.log)、[完整转换回归](pass_regression.log)、
[7项编译检查](compile_checks.log)、[设备最终日志](device/run.log)、[JUnit](device/results.xml)。
原始 trace：[AIC](traces/local_aic_nested.json)、[AIV](traces/local_aiv_nested.json)、
[MIX](traces/mixed_chain_nested.json)；转换后的 PTO IR 与核函数在 `generated/`。

## 尝试过程

1. 新结构对照先在旧版本复现失败。重建后首次检查仍载入未安装的新扩展，安装本工作树产物后通过；
   没有复制其他工作树的二进制。
2. 初版缺失/重复 Wait 负例漏用 block index，被 DSL 在更早阶段拒绝；修正夹具后，PTOAS 确认边界非法。
   汇总异常只展示第一条错误，最终测试同时核对完整报告中的边界诊断。
   保留[夹具错误日志](negative.log)和[错误匹配修正前日志](negative_boundaries.log)，它们不是功能通过证据。
3. `task_20261012_072257_38474109882`：MIX已通过，纯AIC/AIV夹具中的字符串常量不受 DSL 支持，
   停在解析阶段；任务退出1并释放卡。[首次设备任务日志](device_first/run.log)。
4. 将选择量在函数工厂外转为布尔常量；MIX工厂保留原10参数接口，解决测试代码的参数数量检查。
   `task_20261012_072527_394258021621`：3项设备检查全部通过，退出0并释放卡。
   Ruff与格式、diff检查通过。以上失败均没有绕过安全校验或放宽数值检查。

## 复现

使用 `versions.json` 的私有配套工具链，应用 `pypto.patch` 后在本工作树重建并安装扩展。
先 source 私有环境；CPU检查：

```bash
cd "$SCALAR_OVERLAP_ROOT/pypto"
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m pytest tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py -q
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
  tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
  -k 'rejects_pre_wait or requires_one_wait or nested_wait' --platform a2a3 --codegen-only -q
```

[单卡脚本](repro/run_device.sh)接受私有工作树根目录作为第一个参数；通过 `task-submit --device auto`
提交，卡号只读 `TASK_DEVICE`，不要绕过队列。输出默认保存在私有环境的
`artifacts/nested_branch_wait_repro/device/`。

一般只读 GM 前缀的依赖/别名证明、循环边界和通知资源耗尽后的窗口仍需后续处理，当前 goal 未标完成。
