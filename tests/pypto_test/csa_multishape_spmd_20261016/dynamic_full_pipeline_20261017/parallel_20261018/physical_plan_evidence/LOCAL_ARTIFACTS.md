# 本地大产物清单

只记实际路径与字节数，不做hash。此handoff目录不包含二进制或大快照；全部逐文件路径/bytes见 LOCAL_ARTIFACTS.json。

| 根路径 | 文件数 | 字节数 |
|---|---:|---:|
| `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/cpu_v1` | 129 | 86190542 |
| `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/device_probe_build_v1` | 9 | 1083044 |
| `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/device_probe_cache_v1` | 0 | 0 |
| `/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/device_probe_v1` | 1 | 2896 |
| `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/physical_plan_h131072_b16_v1` | 347 | 39541035 |
| `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/physical_plan_h131072_b16_v2` | 517 | 63630134 |

旧v1在候选编译期主动终止，没有整层数值/性能结果。v2为真实完成任务；其中source_matrix目录保存15份三方5窗原始泳道与JSON，源主日志和图保持原处。
