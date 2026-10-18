# incore-profiling 的 msprof timeout 单位与 CANN 不一致

在2026-10-19核对的main，`.claude/skills/incore-profiling/incore_profile.py`
将`--msprof-timeout`描述为秒，默认180；但直接传给CANN的`--timeout`，
后者单位是分钟。同一变量又被用作Python子进程的秒级超时，导致内外时限
不一致，较长的有效模拟可能先被Python中止。

## 无需NPU的复现

环境：a2a3，CANN 9.2.0-beta.2，配套msprof/msopprof。

```bash
source /path/to/cann-9.2.0-beta.2/set_env.sh
msprof op simulator --help
rg -n 'msprof-timeout|msprof_timeout' .claude/skills/incore-profiling/incore_profile.py
```

CANN帮助明确给出`--timeout=<MINUTES>`，范围1～2880。
当前Python代码则是：

```python
prof_group.add_argument(
    "--msprof-timeout", type=int, default=180,
    help="msprof simulator timeout seconds",
)
# collect_cmd:
f"--timeout={args.msprof_timeout}"
# collection和export的Python subprocess.run:
timeout=args.msprof_timeout + 120
```

例如`--msprof-timeout 10`会让CANN允许应用执行10分钟，但Python在130秒
便中止调用；默认180则分别变成CANN 180分钟、Python 300秒，帮助又称180秒。
本问题仅依据命令行合同及当前源码即可复现，无需执行长时间模型或占用NPU。

## 建议修改

一种兼容现有秒级CLI描述的方案：只在CANN参数边界向上取整为分钟，外层
保护用换算后的真实分钟预算，另加导出余量；同时校验范围并明确记录单位。

```python
timeout_seconds = args.msprof_timeout
if not 0 < timeout_seconds <= 2880 * 60:
    raise ValueError("--msprof-timeout must be in (0, 172800] seconds")
simulator_minutes = (timeout_seconds + 59) // 60
collect_cmd.append(f"--timeout={simulator_minutes}")
subprocess_timeout_seconds = simulator_minutes * 60 + export_grace_seconds
```

也可以显式拆成`--simulator-timeout-minutes`和`--collection-timeout-seconds`，
但需要同步skill文档、示例及调用者，避免已有参数默默改变含义。export可能
还需独立预算；任务队列时限也应大于整个采集/导出保护预算。

建议轻量回归仅检查参数转换和传给`run_cmd`的值，不实际sleep或启动NPU：
覆盖10秒、60秒、61秒、默认180秒、边界与非法值；确认CANN分钟数及Python
秒数一致，不让多核大算子的有效采集被意外的更短外层时限中止。

## 本次观察的边界

我们在完整CSA的24 AIC+48 AIV回放中发现该单位问题；相同独立二进制在
真机已通过完整物理输出比较。本次一次采集最终触发的是本地另设的25分钟
队列上限，不能将该次终止直接归因于上述上游Python时限。这里报告的是
独立可核对的工具单位错误，不声称PyPTO算子或CANN数学实现有缺陷。

本次只提issue及修改建议，不提交PR、不添加标签。
