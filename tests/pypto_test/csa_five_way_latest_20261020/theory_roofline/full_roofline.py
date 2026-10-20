# SPDX-License-Identifier: Apache-2.0
"""复算完整 CSA 的异构资源 roofline、检查来源并生成可交接报告。

不同资源使用不同工作单位，统一换算为峰值服务时间后取最大值，避免把
INT8矩阵运算、FP32 exp、Top-K比较和字节流量相加成虚假的总TFLOPS。
这份报告覆盖全部计算阶段的资源模型，并明确保留缓存、编译展开和依赖
开销的未知量；它不是周期精确的端到端模拟器，也不启动设备测试。
"""

import ast
import csv
import json
from pathlib import Path
import subprocess

import analyze as cube
import vector_model as vec
import memory_model as mem
import vector_sort_bound

HERE, FIVE, REPO = cube.HERE, cube.FIVE, cube.REPO
FORMAL = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def source_audit():
    """对实测快照做字节比较和业务调用覆盖核查，不生成hash清单。"""
    files, changed = [], []
    for source in sorted((FIVE/"source_dynamic").glob("*.py")):
        if source.read_bytes() != (FORMAL/source.name).read_bytes():
            changed.append(source.name)
        files.append(source.name)
    assert len(files) == 37 and not changed, changed
    helper = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/rmsnorm.py"
    old = subprocess.check_output(["git", "-C", str(REPO), "show", "13259f92:"+helper])
    assert old == (REPO/helper).read_bytes(), "共享RMS helper已偏离实测版本"
    isa_reference = REPO.parent / "pto-isa-src/include/pto/npu/a2a3"
    isa_managed = REPO.parent / (
        ".worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3")
    isa_equal = {}
    for name in ("TUnaryOp.hpp", "TRowReduceOps.hpp"):
        isa_equal[name] = (isa_reference/name).read_bytes() == (isa_managed/name).read_bytes()
        assert isa_equal[name], f"当前Simpler管理的{ name }与建模参考不一致"
    functions = {}
    for file in (FIVE/"source_dynamic").glob("*.py"):
        for node in ast.parse(file.read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert node.name not in functions
                functions[node.name] = (file.name, node)
    reachable, todo = set(), ["vector_schedule"]
    while todo:
        name = todo.pop()
        if name in reachable:
            continue
        reachable.add(name)
        todo.extend(call.func.id for call in ast.walk(functions[name][1])
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in functions and call.func.id not in reachable)
    compute_files = {
        "hc_stages.py", "qkv_stages.py", "query_projection_stages.py",
        "kv_stages.py", "compressor_pool_stages.py", "compressor_output_stages.py",
        "indexer_compressor_output_stages.py", "indexer_cache_stages.py",
        "indexer_query_stages.py", "indexer_score_stages.py", "indexer_topk_stages.py",
        "indexer_publish_stages.py", "indexer_group_stages.py",
        "indexer_scale_layout_stages.py", "attention_plan_stages.py",
        "attention_stream_stages.py", "output_projection_a_stages.py", "output_projection_b_stages.py",
    }
    control_files = {
        "phase_sync.py": "通知、轮询、fence；不作为零成本Vector指令",
        "signals.py": "通知、轮询、缓存维护；吞吐表不提供这些指令的roofline",
        "attention_dependencies.py": "前驱owner等待和地址推导",
        "request_geometry.py": "Scalar请求边界/位置判定",
        "metadata_stages.py": "Scalar前缀、紧凑行地址及MTE取行；RoPE Tile初始化计入两套压缩norm",
        "cache_stages.py": "原始KV写回；数据搬运计入字节账本",
        "indexer_scale_plan_stages.py": "Scalar页owner链、FP16单值更新、64B页搬运",
        "vector_prefix_stages.py": "压缩/原KV业务helper的编排；无新增数学",
        "vector_schedule.py": "48核分工和依赖编排",
    }
    scalar_helpers = {
        "indexer_decode_group", "indexer_group_plan_stage", "indexer_leaf_range",
        "indexer_long_leaf_plan", "indexer_query_visibility", "indexer_work_item",
        "indexer_scale_layout_base", "indexer_visible_root_count",
        "attention_query_geometry", "physical_plan_supported",
    }
    covered = []
    for name in sorted(reachable):
        file, node = functions[name]
        assert file in compute_files or file in control_files, (file, name)
        treatment = ("数学/Vector metadata所属阶段已纳入工作量模型"
                     if file in compute_files else control_files[file])
        if name in scalar_helpers:
            treatment = "Scalar动态metadata；未冒充Vector算术，时长属于端到端预测缺口"
        if name == "hc_projection_piece":
            treatment = "HC_CONSUMER_REDUCE分支；本次七档均关闭，运行的是hc_reduce_range"
        covered.append(dict(function=name, file=file, line=node.lineno, treatment=treatment))
    # 非zero Cube/Vector阶段有源码锚点；不能由一个空的“已覆盖”标记代替计算。
    for sources in vec.STAGE_SOURCES.values():
        for source in sources:
            if "::" in source:
                file, name = source.split("::")
                assert name in functions and functions[name][0] == file, source
    return dict(
        source_snapshot="source_dynamic，实测13259f92",
        formal_files_equal=files, rms_helper_equal=True,
        isa_reference_directory=str(isa_reference), isa_managed_directory=str(isa_managed),
        isa_headers_byte_equal=isa_equal,
        vector_reachable_functions=covered, vector_stage_count=len(vec.STAGE_SOURCES),
        scope="覆盖每个业务计算阶段；Scalar等待与编译生成的额外指令明确不作精确时长预测",
    )


def combine(row, vector, ports, memory):
    c = ports["current_assignment_us"]["cube_cycles"]
    vlow, vhigh = vector["ideal_reduction_current_assignment_us"], vector["current_assignment_max_us"]
    pb = ports["current_assignment_us"]["right_bytes"]
    other_ports = max(ports["current_assignment_us"].values())
    hot = max(c, vlow, other_ports, memory["hot_start_capacity_floor_us"])
    hot_generic = max(c, vhigh, other_ports, memory["hot_start_capacity_floor_us"])
    cold_min = max(hot, memory["cold_read_max_reuse_us"])
    cold_scenario = max(hot_generic, memory["cold_read_writeback_per_query_us"])
    actual = row["actual_dynamic_l1_us"]
    return dict(
        history=row["history"], batch=row["batch"], query_tokens=row["query_tokens"],
        cube_ideal_balance_us=row["issued_cube_us"], cube_current_assignment_us=c,
        vector_ideal_reduction_us=vlow, vector_generic_reduction_max_us=vhigh,
        vector_generic_reduction_min_us=vector["current_assignment_min_us"],
        l0b_current_assignment_us=pb,
        hbm_cold_max_reuse_read_us=memory["cold_read_max_reuse_us"],
        hbm_cold_per_query_read_writeback_us=memory["cold_read_writeback_per_query_us"],
        # 用户要求的串行对照：表内四项完全不重叠时求和。
        # 使用未取整原值；这不是阶段DAG串行模拟，也不含尚未建模的控制开销。
        serial_table_sum_min_us=c+vlow+pb+memory["cold_read_max_reuse_us"],
        serial_table_sum_max_us=c+vhigh+pb+memory["cold_read_writeback_per_query_us"],
        hot_capacity_resource_floor_us=hot, hot_generic_resource_model_us=hot_generic,
        cold_resource_floor_us=cold_min, cold_per_query_writeback_model_us=cold_scenario,
        actual_dynamic_l1_us=actual, actual_over_hot_resource_floor=actual/hot,
        actual_over_cold_resource_floor=actual/cold_min,
        gap_to_hot_resource_floor_us=actual-hot,
        theoretical_peak_headroom_fraction=1-hot/actual,
        memory_service_sensitivity_us={
            rate: max(hot, duration) for rate, duration in memory["gm_aggregate_rate_sensitivity_us"].items()},
        # 峰值占用模型不是PMU引擎利用率。
        cube_peak_equivalent_fraction=c/actual, vector_peak_equivalent_fraction=vhigh/actual,
        ports=ports, memory=memory, vector=vector,
        observed_phase_probe=row.get("stage_probe"),
    )


def case_name(row):
    return f"{row['history']//1024}K/B{row['batch']}"


def report_markdown(report):
    rows = report["rows"]
    lines = [
        "# CSA 完整计算链的 Cube、Vector 与内存 roofline",
        "",
        "日期：2026-10-20。仅离线复算已有七档，不新增设备测试，不修改算子，不push。",
        "",
        "**结论：Vector已补齐，先前只有Cube+HBM的结果不能代表完整CSA。** "
        "当前峰值资源模型仍显示明显余量，但这个余量包含必须的数据搬运、标量控制、"
        "指令启动及同步，不能全部记成可以兑现的优化收益。",
        "",
        "此处“完整”指覆盖HC_pre、输入RMS、Q/KV/两套压缩、Indexer/Top-K、稀疏Attention、"
        "O投影及HCpost的计算工作量与主要数据通路；**不是周期精确的端到端时延预测**。"
        "未知的L2实际吞吐、HBM命中率和生成代码额外指令保持为未知，不填经验利用率伪装实测。",
        "",
        "## 1. 硬件与计量口径",
        "",
        "本机为Ascend910_9392，一个逻辑device对应一个die；24 AIC、48 AIV，"
        "同轮Profiler记录AIC/AIV均1850MHz，L2为192MiB。未使用双die整卡峰值。",
        "",
        "| 资源/操作 | 峰值口径 |",
        "| --- | ---: |",
        "| Cube BF16/FP16 | 363.7248 TFLOP/s |",
        "| Cube INT8 | 727.4496 TOP/s |",
        "| Cube标准FP32 | 90.9312 TFLOP/s；HF32另有模式 |",
        "| Vector FP32 Add/Mul/Min/Max | 单AIV 64元素/cycle；全48核5.6832万亿元素操作/s |",
        "| Vector FP32 Exp/Sqrt/Div | 单AIV 32元素/cycle；全48核2.8416万亿元素/s |",
        "| Vector Sort32/MrgSort | 单AIV 2排序项/cycle；全48核1776亿排序项/s |",
        "| HBM | 每die 1.6 TB/s规格上限，非本机实测持续值 |",
        "| L1→L0A / L1→L0B | 每AIC 256 / 128 B/cycle |",
        "| Score FIX INT32→FP16→L1 | 每AIC 64元素/cycle，即128 B/cycle |",
        "",
        "来源：[A3/2201架构](https://asc.gitcode.com/guide/programming_guide/advanced_programming/"
        "hardware_implementation/architecture_spec/npu_arch_2201.html)、"
        "[Cube官方表](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html)、"
        "[Vector官方表](https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html)、"
        "[Huawei CloudMatrix384 §3.3.1](https://arxiv.org/html/2506.12708v2#S3.SS3.SSS1)。",
        "",
        "不能把FP32向量元素操作、特殊函数和排序项都称作FLOP。FMA若按两个FLOP计，"
        "与两条独立Mul/Add的吞吐也不同；本代码保留原有舍入顺序，没有假设自动融合FMA。",
        "FP32 HC投影按标准Mmad模式计算，历史敏感性账本仍保留，未改其精度。",
        "",
        "## 2. 如何组合这些不同资源",
        "",
        "```text",
        "Cube服务量 = Σ(各精度矩阵运算量 / 对应Cube峰值)",
        "Vector服务量 = Σ(各指令类别处理元素数 / 该类每核吞吐 / 频率)",
        "端口服务量 = 经过该端口的字节数 / 端口峰值",
        "内存服务量 = 明确缓存条件下的HBM字节数 / HBM带宽",
        "",
        "完整调用资源下界 ≥ max(最忙AIC的Cube服务量, 最忙AIV的Vector服务量,",
        "                         各片上端口服务量, 对应条件下的HBM服务量)",
        "```",
        "",
        "先按真实核号累加再取最大值，不能把各阶段最忙核相加，也不能用全核平均"
        "掩盖固定分工。Cube与Vector、计算与DMA可重叠，不能将整层服务时间简单相加。"
        "真实数据依赖、单次指令延迟、轮询、FFTS以及有限缓冲会进一步抬高可达时延。",
        "",
        "## 3. 七档结果",
        "",
        "单位μs；所有S6、TP1、正式W8A8第4层、四主投影权重NZ。"
        "实测来自同日五口径数据中已冻结的动态版本13259f92，"
        "与37个正式源码文件及共享RMS helper核对一致。其它会话随后改过的多task基线不混入。",
        "",
        "Vector范围是理想归约与通用Tile归约/末级merge处理量的模型敏感性，"
        "不是实测波动或所有可能实现的严格上下界。冷HBM两端分别是假设最大KV复用的"
        "只读冷供给，以及逐query供给并在窗口内完成外部输出写回。",
        "",
        "| 档位 | Cube最忙核 | Vector服务模型 | L1→L0B最忙核 | 冷HBM场景 | 串行总时间 | 动态实测L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {case_name(r)} | {r['cube_current_assignment_us']:.2f} | "
                     f"{r['vector_ideal_reduction_us']:.2f}～{r['vector_generic_reduction_max_us']:.2f} | "
                     f"{r['l0b_current_assignment_us']:.2f} | {r['hbm_cold_max_reuse_read_us']:.2f}～"
                     f"{r['hbm_cold_per_query_read_writeback_us']:.2f} | "
                     f"{r['serial_table_sum_min_us']:.2f}～{r['serial_table_sum_max_us']:.2f} | "
                     f"{r['actual_dynamic_l1_us']:.2f} |")
    lines += [
        "",
        "**串行总时间 = Cube + Vector + L1→L0B + 冷HBM**，按表内四项完全不重叠求和，"
        "合计使用未取整原值。它是资源串行情景对照，不是当前代码的实测或完整阶段DAG"
        "串行模拟；仍未计入Scalar、通知、指令启动等尚未定量的开销，不能称为真实耗时上界。",
        "",
        "下面取资源最大值。热起始一列允许起始L2中最多已有192MiB只读输入；仍受Cube、"
        "Vector及片上端口限制。冷场景一列增加外部输入首次供给。两者均没有将未知L2"
        "聚合带宽、完整依赖图和Scalar延迟伪装成已知，所以都是乐观资源口径。",
        "",
        "| 档位 | 热起始资源下界 | 冷场景资源模型 | 实测÷热资源下界 | 实测与热下界之差 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {case_name(r)} | {r['hot_capacity_resource_floor_us']:.2f} | "
                     f"{r['cold_resource_floor_us']:.2f}～{r['cold_per_query_writeback_model_us']:.2f} | "
                     f"{r['actual_over_hot_resource_floor']:.2f}× | {r['gap_to_hot_resource_floor_us']:.2f} |")
    lines += [
        "",
        "**最后一列只是相对乐观下界的差额，不是承诺可省的时间。** "
        "例如不能据此声称802μs能直接优化到170μs；矩阵、排序、KV搬运要按依赖供给，"
        "短Tile启动、Scalar寻址、UB/L1冲突和同步都会占用时间。",
        "",
        "此前B16/B28的Cube全24核平均为161.77/287.07μs；本表改用实际最忙AIC，"
        "所以更高。QB实际分给20核且logical=physical%20，KV另有4核循环置换；"
        "不是新增计算，也不是性能实测变了。15个矩阵阶段由独立循环模型交叉核对，总运算量完全相符。",
        "",
        "## 4. Vector具体算了什么",
        "",
        "共26个业务工作量分组，覆盖实际调用链的全部计算阶段：",
        "",
        "- HC加宽、归约、sigmoid门控、20轮Sinkhorn、四路混合、输入RMS。",
        "- QR归一化量化，主Query反量化/RMS/RoPE，原始KV归一化/RoPE。",
        "- 两套压缩pool的max/exp/加权合并、state更新、norm/RoPE、compact量化。",
        "- Indexer query反量化/RoPE/Hadamard后量化，head系数、scale处理、叶内/跨叶Top-K。",
        "- 页连续性证明、选中页物理地址与有效块计划、softmax、在线归并及逆RoPE。",
        "- O_A输出量化、八组O_B反量化左折叠、四路HCpost。",
        "",
        "注意三项容易漏算的成本：归约不能直接按Add64元素/cycle算；"
        "带tmp的高精度TRSQRT在当前A3实现为vsqrt+vdiv；"
        "Sort32/MrgSort按(score,index)项计数，不能把1024个FP32暂存误当1024个排序项。"
        "短8K采用G2候选半叶分工，已单独建模，没有复用长G6按query分工的公式。",
        "排序计量单位同时参照[官方Sort32/MrgSort接口说明](https://www.hiascend.com/"
        "document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html)"
        "及[CANN DeepSeekV4的A3 Top-K理论周期示例](https://gitcode.com/cann/"
        "cann-recipes-infer/blob/master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md)。",
        "",
        "以下是B16/B28各阶段最忙AIV的通用归约服务模型；各行可能是不同核且阶段可重叠，**不能相加成整层时延**。",
        "",
        "| Vector阶段 | 128K/B16 | 128K/B28 |",
        "| --- | ---: | ---: |",
    ]
    b16 = next(r for r in rows if r["history"] == 131072 and r["batch"] == 16)
    b28 = next(r for r in rows if r["history"] == 131072 and r["batch"] == 28)
    for a, b in zip(b16["vector"]["stages"], b28["vector"]["stages"]):
        assert a["name"] == b["name"]
        lines.append(f"| {a['name']} | {a['current_assignment_min_us']:.3f} | {b['current_assignment_min_us']:.3f} |")
    lines += [
        "",
        "## 5. 内存不是只有HBM",
        "",
        "大矩阵权重合计178.29MB，已接近192MiB L2容量；长历史key/scale、KV以及中间结果"
        "会进一步争用缓存。GM地址访问可能命中L2，不能全部除以1.6TB/s；也不能假定全部命中。",
        "HBM账本已补上norm/scale/APE、旧state、RoPE与metadata；只统计实际访问区域，"
        "不按整张cache/state allocation大小收费。本轮生成的新KV不当作外部冷读。",
        "",
        "语义GM账本统计当前必需的物化和直接消费，B16/B28分别约"
        f"{b16['memory']['semantic_gm_ledger_total_bytes']/1e6:.2f}/"
        f"{b28['memory']['semantic_gm_ledger_total_bytes']/1e6:.2f}MB。"
        "它未包含后端重复Tile搬入、缓存行放大、信号和临时溢出，不能当精确动态事务量。",
        "没有取得本次整层实际HBM/L2流量，因此对GM总服务率做以下敏感性分析：",
        "",
        "| 档位 | 若GM总服务率2TB/s | 4TB/s | 7TB/s | 当前完整L1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        sens = r["memory_service_sensitivity_us"]
        lines.append(f"| {case_name(r)} | {sens['2']:.2f} | {sens['4']:.2f} | {sens['7']:.2f} | {r['actual_dynamic_l1_us']:.2f} |")
    lines += [
        "",
        "表中已与Cube/Vector/片上端口取max；这些带宽不是本机实测值。7TB/s仅借用"
        "[官方192MB L2示例中的假设](https://www.hiascend.com/document/detail/zh/"
        "CANNCommunityEdition/83RC1alpha001/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0036.html)，"
        "不能把该示例当成9392保证规格。"
        "[官方性能指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/"
        "83RC1alpha003/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_00014.html)"
        "也要求把共享带宽上的读写合计，并提醒小DMA达不到峰值。",
        "",
        "因此，B16/B28如果GM有效服务率仅2TB/s，当前数据流即使忽略等待也要约"
        f"{b16['memory_service_sensitivity_us']['2']:.0f}/"
        f"{b28['memory_service_sensitivity_us']['2']:.0f}μs；"
        "这比只看Cube算术更能说明为什么不能把全部差额看成调度损失。",
        "",
        "## 6. 与真机阶段证据对照",
        "",
        "同版本已有72核阶段探针给出：",
        "",
        "| 阶段 | B16阶段区间 | B28阶段区间 | 资源解释 |",
        "| --- | ---: | ---: | --- |",
        "| Score AIC / AIV | 192.64 / 181.67 | 329.82 / 324.70 | Cube、排序、MTE1、FIX均需考虑 |",
        "| Attention AIC / AIV | 153.47 / 158.41 | 304.19 / 304.77 | 计算工作小于数据交接及实际流水区间 |",
        "| 主QB AIC | 60.57 | 88.91 | 20核列条带、权重供给和短矩阵效率 |",
        "",
        "这些探针区间扣掉外层GM依赖等待，但仍含内部READY/ACK、DMA、Scalar和探针；"
        "它们不是纯引擎忙碌时间。旧独立Score PMU及修改过退避的模拟流水没有作为本轮整层实测。",
        "",
        "Score最忙核的Cube下界B16/B28为91.33/170.21μs，"
        "仅排序归并约80～82/148～150μs，QK经FIX转L1再参与WS也受片上端口限制。"
        "两边算力接近，所以单独加速Cube可能转成Vector受限；不能忽略任一边。",
        "",
        "Attention的Cube约22.14/38.75μs，Vector主要算术约21/37μs；"
        "但当前QK/PV/概率/KV环槽主要GM流量已达374.34/655.10MB。"
        "这支持优先审视数据交接、DMA粒度与流水重叠，"
        "不能仅根据Cube低利用率继续加大矩阵流水。",
        "",
        "## 7. 能得出与不能得出的结论",
        "",
        "能得出：完整CSA未逼近纯Cube/Vector算力资源上限；长历史Score已同时受到"
        "Cube、排序和片上转运约束；Attention有明显的数据交接与流水成本。"
        "当前核分工约束下，理论资源差额仍大，但它是乐观余量上限。",
        "",
        "不能得出：某个精确微秒值已经可达，或还能确定省掉某个百分比。"
        "要把资源下界收紧为可达预算，需要同版本整层HBM/L2事务与有效带宽、"
        "Vector/Cube/FIX/MTE忙碌时间，以及未改通信语义的关键路径证据。"
        "本次没有因此擅自启动新一轮设备测试或优化。",
        "",
        "## 8. 复现及文件",
        "",
        "```bash",
        "python tests/pypto_test/csa_five_way_latest_20261020/theory_roofline/full_roofline.py",
        "```",
        "",
        "- `full_roofline.py`：组合模型、源码覆盖核查、JSON/CSV/本文生成。",
        "- `vector_model.py`：26个Vector工作分组、各类别吞吐、48核分配。",
        "- `vector_sort_bound.py`：长G6与短G2逐叶排序计数。",
        "- `memory_model.py`：24核矩阵/片上端口、冷热HBM及GM语义账本。",
        "- `analyze.py` / `ANALYSIS.json`：15个Cube阶段的有用/执行运算量组件。",
        "- `FULL_ROOFLINE.json`：七档资源摘要、假设、来源和详细组件文件引用。",
        "- `FULL_ROOFLINE.csv`：七档扁平表，便于下载后比较。",
        "- `VECTOR_MODEL.json` / `MEMORY_MODEL.json`：两个资源组件的独立导出。",
        "- `SOURCE_COVERAGE.json`：37文件字节一致、共享RMS helper及可达函数审计。",
        "- `INITIAL_CUBE_ONLY.md`：历史不完整版本，保留过程，禁止当最新结论。",
        "",
        "核查包括15个矩阵阶段独立计数守恒、既有Score逐核块数真机对照、"
        "两种Top-K路径有效候选覆盖及Softmax/query/HCpost元素守恒。"
        "没有下载权重、提交二进制或生成hash清单。",
        "",
    ]
    return "\n".join(lines)


def main():
    cube.main()
    vector_sort_bound.main()
    vec.main()
    source = source_audit()
    base = cube.read(HERE/"ANALYSIS.json")
    vectors = cube.read(HERE/"VECTOR_MODEL.json")
    rows, memory_rows = [], []
    for row, vector in zip(base["rows"], vectors["rows"]):
        ports = mem.cube_ports(row)
        memory = mem.memory(row, vector)
        rows.append(combine(row, vector, ports, memory))
        memory_rows.append(dict(history=row["history"], batch=row["batch"], ports=ports, memory=memory))
    report = dict(
        status="complete_heterogeneous_resource_roofline_with_explicit_limits",
        exact_end_to_end_latency_prediction=False,
        vector_compute_stages_covered=26,
        source=source, hardware=base["hardware"],
        vector_rates=vec.RATES,
        sources=[vec.VECTOR_SOURCE, mem.CUBE_SOURCE,
                 "https://arxiv.org/html/2506.12708v2#S3.SS3.SSS1"],
        assumptions=[
            "七档S6、无dummy；对应已有实测输入及13259f92源码。",
            "峰值服务模型忽略指令头、Scalar、同步延迟和额外编译展开；非可达时间承诺。",
            "Vector归约给理想/通用实现敏感性，末级merge给exhausted处理量范围。",
            "HBM冷/热条件明确分开，GM总请求不等于HBM事务。",
            "L2实际聚合带宽未知，以2/4/6/7TB/s做条件分析，不标成实测规格。",
            "serial_table_sum是表内Cube、Vector、L0B、冷HBM的无重叠算术和，不是完整串行执行时延。",
            "数学与输入保持不变；没有新的NPU运行、算法优化或push。",
        ],
        rows=rows,
    )
    # 详细逐核计数保存在组件JSON；总报告只保留摘要，避免重复提交数MB数组。
    compact = dict(report)
    compact_rows = []
    for row in rows:
        summary = {k: v for k, v in row.items() if k not in ("ports", "vector")}
        summary["ports_summary"] = {k: v for k, v in row["ports"].items()
                                    if k not in ("counts_per_aic", "service_us_per_aic", "stages")}
        summary["vector_summary"] = {k: v for k, v in row["vector"].items()
                                     if k not in ("stages", "sorting")}
        summary["component_details"] = ["VECTOR_MODEL.json", "MEMORY_MODEL.json"]
        compact_rows.append(summary)
    compact["rows"] = compact_rows
    for name, data in (
        ("FULL_ROOFLINE.json", compact),
        ("SOURCE_COVERAGE.json", source),
        ("MEMORY_MODEL.json", dict(source=mem.CUBE_SOURCE, rows=memory_rows)),
    ):
        (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    (HERE/"FULL_ROOFLINE.md").write_text(report_markdown(report))
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, str))]
    with (HERE/"FULL_ROOFLINE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in scalar_keys} for r in rows)
    for r in rows:
        print(case_name(r), "cube/vector/port/hot/cold/actual",
              *(round(r[k], 2) for k in ("cube_current_assignment_us", "vector_generic_reduction_max_us",
                                        "l0b_current_assignment_us", "hot_capacity_resource_floor_us",
                                        "cold_resource_floor_us", "actual_dynamic_l1_us")))


if __name__ == "__main__":
    main()
