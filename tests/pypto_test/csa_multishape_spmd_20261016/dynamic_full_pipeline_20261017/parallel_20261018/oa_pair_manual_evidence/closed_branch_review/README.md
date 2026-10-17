# 正式迁移前：O_A pair关闭分支审计（2026-10-18）

**结论：关闭候选后恢复原行为的源码与生成证据齐全。** 冻结的668参考/候选33源未修改；只做一次OA256/OB128/QAfalse完整CPU，不增加设备任务。

## 源级可达性与Host合同

`check_source_closure.py` 对cube/vector两个函数分别进行有限constexpr折叠：OA256且组流水开启、OA128且组流水关闭、OA256且组流水关闭，以及手工OA128/T97的防御性回退，共8个函数AST比较全部与668原版相同。只消去已知条件，不修改未知device表达式，不将真实T冻结到某个值。

新pair数学入口和五owner等待仅有一处调用，各自受完全相同祖先条件约束：

```text
OUTPUT_GROUP_PIPELINE && OA_COLS == 128 && 0 < tokens <= 96
```

正式Host原策略仍是T≤96选N128；`policy.py`逐字未改，其字段/默认值/constexpr映射全部保持。`entry.py`、`binding.py`、`runtime_abi.py`、`signals.py`、`wait_codegen.py`也逐字未改，因此没有增加runtime root、工作区、phase或更换信号协议。T1..96只一个M128块；完整pair/single唯一写和五owner集合已由`../owner_evidence/rotated_owner_model.json`覆盖，T97+不进入候选。

## 唯一关闭分支CPU

`cpu_oa256/result.json`：完整PASS；130 roots、58动态轴、single MIX、sync_start；没有提供样例shape。controls为long G6/64/1024、PREFETCH1、BALANCED/ROOTS1、OA256/OB128、组流水与HC行依赖开启、ATTN_META/HC_CONSUMER_REDUCE关闭、QA_M16_PARALLEL=false。

参考不是B-only候选，而是同控旧设备任务的原SPMD reference：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/query_linear_h131072_b24_v1/build_output/decode_csa_spmd_5e_gzb5p`

其角色和controls由同轮`source_matrix/result.json`核实。旧参考是9d可达源码；以QA=false消去后来新增分支后，32个设备源码模块的AST与本候选668参考相同。旧Host policy不用于证明相同，单独以同controls和候选policy对668逐字不变限定范围。没有把B-only改过的Q_B当作同基线。

## 已生成物比较

| 项目 | 结果 |
|---|---|
| 整个PTO（包括Q_B与AIV） | 归一化后逐字相同；只去loc/SSA名及常量声明顺序，保留运算、类型、数值、cache策略和物理地址 |
| AIC Tile分配 | 553项address/space/dtype/rows/cols/size全部相同 |
| AIC空间地址上界 | Mat475136、Left65536、Right65536、Acc131072、Vec16384 B，与参考相同 |
| AIC局部事件 | 1700条set_flag/wait_flag/pipe_barrier，顺序及event id相同 |
| AIV局部事件 | 3088条，顺序及event id相同 |
| 新pair与五owner等待 | OA256 PTO内不可达、未残留 |

审计结果`closed_codegen_audit.json`；三个`.diff`文件均为空。复现脚本`audit_closed_codegen.py`只读已经存在的PTO/C++，复用已归档审计器的纯文本归一化函数，不执行其main，也不触发编译。

不声称机器码全文相同：旧设备reference的wait-backoff辅助实现与CPU默认wait不同。上述局部事件比较只截取实际cube/vector计算函数，不混入辅助头文件代码。关闭组流水仅做源级AST证明，未额外增加第二次完整CPU。开启pair的精度、graph、guard、代际和性能仍由主会话设备结果判定。

## 本轮记录

主会话反馈两代表完整PASS：long B4，原task505.98/ref668 555.80/candidate527.14μs，相对参考降低5.157%，5/5窗更快；short B16，676.50/710.28/695.36μs，降低2.101%，5/5窗更快。本子任务未执行或重复这些设备任务。其余受影响目标点long B8/B16以及S1/S5功能由主会话继续处理，这里不提前记通过。

只读审计复现：

```bash
python check_source_closure.py
python audit_closed_codegen.py
```

`run_cpu.sh`与`controls_oa256.json`保存唯一CPU命令；既有`cpu_oa256`拒绝覆盖，不再运行。本地全部产物仅列路径/大小，无hash。大CPU产物留本地，迁移可只归档本目录小文本、result和必要源码。
