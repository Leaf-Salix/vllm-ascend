# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""历史静态 shape 单 SPMD 与原动态 PTO 的比较脚本，不是目标动态版本。"""

import argparse
import ast
import contextlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from shape_specialization import STATIC_ROOT, build_variant, descriptors_from_tensors
from stage_clock import compiler_probe, export_stage_clock
from signal_polling import bypass_signal_wait
from preserve_math_pipeline import preserved_pipeline
from phase_memory import scoped_memory_planning
from stage_codegen import independent_stage_codegen
from workspace_base import workspace_base_codegen
from task_completion import task_counter_codegen
from compact_abi import build_compact_entry
from core_specialization import specialize_core_identity, reuse_identical_core_binaries
from production_sources import SIGNAL_SOURCE

from test_static_jit_device import (
    CompileOptions, PreparedStaticArguments, checks_passed, collect_state,
    decode_additional_config, exact_checks, guard_checks, make_fixture, make_layer,
    native_session, passes, progress_expectation, pypto, torch, torch_npu, verify_isolation,
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def statistics_us(values):
    ordered = sorted(values)
    return {"samples_us": values, "count": len(values), "median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "min_us": ordered[0], "max_us": ordered[-1],
            "p95_us": ordered[math.ceil(0.95 * len(values)) - 1]}


def export_window(source, destination, side):
    """保留原始六列；第 4/5 列是 incore 起止，不能把收任务时间作为开始。"""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("window_*"))
    raw = json.loads((destination / "chip_swimlane_records.json").read_text())
    frequency = raw["metadata"]["clock_freq_hz"]
    rows = raw["aicore_tasks"]
    if raw["chip_swimlane_level"] != 1 or not rows or frequency <= 0:
        raise ValueError("需要有效 L1 核内任务时间记录")
    if any(len(row) != 6 or row[4] < row[3] or row[3] <= 0 for row in rows):
        raise ValueError("原始泳道时间戳异常")
    if len({row[0] for row in rows}) != 72:
        raise ValueError("泳道未覆盖 24 AIC + 48 AIV")
    if side == "single_spmd" and (len(rows) != 72 or len({row[1] for row in rows}) != 1):
        raise ValueError("单 SPMD 不应产生多个 incore task")
    tables = list(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("*/kernel_config.py"))
    names = {}
    for table in tables:
        if (side == "single_spmd") != ("decode_csa" not in table.parent.name):
            continue
        tree = ast.parse(table.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KERNELS"
                                                    for t in node.targets):
                for item in node.value.elts:
                    values = {ast.literal_eval(k): v for k, v in zip(item.keys, item.values)}
                    names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
        shutil.copy2(table, destination / (table.parent.name + "_kernel_config.py"))
    save(destination / "name_map.json", {"callable_id_to_name": names})
    converted = subprocess.run(
        [sys.executable, "-m", "simpler_setup.tools.swimlane_converter",
         str(destination / "chip_swimlane_records.json"), "--func-names",
         str(destination / "name_map.json"), "-o", str(destination / "swimlane.json")],
        text=True, capture_output=True)
    (destination / "converter.log").write_text(converted.stdout + converted.stderr)
    converted.check_returncode()
    return {"incore_span_us": (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / frequency,
            "clock_freq_hz": frequency, "core_records": len(rows),
            "task_count": len({row[1] for row in rows}), "cores": len({row[0] for row in rows}),
            "raw": str(destination / "chip_swimlane_records.json"),
            "trace": str(destination / "swimlane.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--range-policy", action="store_true",
                        help="由工作量区间选择策略，不叠加手工调优开关")
    parser.add_argument("--hc-compact-stream", action="store_true",
                        help="独立候选：在正式区间入口上将HC前段改为紧凑GM双槽交接")
    parser.add_argument("--metadata-replay-only", action="store_true",
                        help="仅验固定地址的长度/页表A→B→C→A；不重复性能采样")
    parser.add_argument("--metadata-history", type=int, action="append",
                        help="metadata重放的额外历史长度，先CPU检查同策略；可重复")
    parser.add_argument("--paired-selected-spmd", action="store_true",
                        help="行依赖功能任务中，附带同进程旧/新单SPMD各3个L1窗口")
    parser.add_argument("--stage-clock", action="store_true", help="仅做归因，插桩结果不用于性能达标")
    parser.add_argument("--internal-wait-clock", action="store_true",
                        help="仅归因Indexer/QK内部Scalar FFTS等待，不参与性能目标")
    parser.add_argument("--qk-phase-clock", action="store_true",
                        help="仅归因QK/PV AIV的KV、softmax、PV合并和输出四段")
    parser.add_argument("--indexer-phase-clock", action="store_true",
                        help="仅归因Indexer AIC页表读取至Key搬运提交的Scalar区间")
    parser.add_argument("--preserve-math-pipeline", action="store_true",
                        help="归因实验：保留选定动态 PTO 数学函数的局部流水")
    parser.add_argument("--balanced-schedule", action="store_true",
                        help="按当前规格重新平衡纯 AIC/AIV 分片，保留数学依赖及 MIX 配对")
    parser.add_argument("--phase-memory", action="store_true",
                        help="实验：每个已排空流水的阶段独立规划片上内存")
    parser.add_argument("--compile-schedule", action="store_true",
                        help="将分档已确定的调度表编入核内控制流")
    parser.add_argument("--precise-payload-cache", action="store_true",
                        help="按各阶段实际 Scalar GM 读写保留业务 DCache 维护")
    parser.add_argument("--schedule-policy", choices=("critical_path", "earliest_ready"), default="critical_path")
    parser.add_argument("--independent-stages", action="store_true",
                        help="归因实验：同一 IR 的 widen 阶段独立 PTOAS 编译后作 device 调用")
    parser.add_argument("--stage-kernels", default="hc_widen_rms",
                        help="独立编译的阶段名，用逗号分隔；* 表示所有阶段")
    parser.add_argument("--compact-stage-inputs", action="store_true",
                        help="仅传入每个内部阶段实际使用的参数，外层 ABI 不变")
    parser.add_argument("--stage-noinline", action="store_true",
                        help="归因：全阶段独立编译后保留 device 调用边界")
    parser.add_argument("--workspace-base", action="store_true",
                        help="归因实验：由已验证的 arena 基址重建临时 GM 指针")
    parser.add_argument("--bypass-wait", action="store_true", help="测试 ld_dev 信号等待，写侧/数据维护不变")
    parser.add_argument("--cached-wait", action="store_true",
                        help="保留 DCCI+cached GM 读取，只调整信号未就绪时的轮询间隔")
    parser.add_argument("--wait-backoff-ticks", type=int, default=0,
                        help="信号未就绪后的 SYS_CNT 等待间隔；当前 50MHz，25 tick 为 0.5us")
    parser.add_argument("--aggregate-completion", action="store_true", help="保持数学依赖偏序的完成通知汇聚")
    parser.add_argument("--task-completion", action="store_true",
                        help="测试原逻辑 task 的 AtomicAdd 完成计数，消费者等待原 DAG 前驱")
    parser.add_argument("--pipe-profile", action="store_true",
                        help="正常计时结束后额外采 PyTorch PipeUtilization，只用于硬件瓶颈归因")
    parser.add_argument("--compact-abi-reference", type=Path,
                        help="诊断：同规格参考编译产物，将临时 GM 参数收敛为一个 arena")
    parser.add_argument("--schedule-cost-profile", type=Path,
                        help="离线调优：使用同规格阶段数学/维护/发布的采样成本重新分配")
    parser.add_argument("--rotate-wait-order", action="store_true",
                        help="初始化时按消费者核旋转等待表，保持完整依赖集合")
    parser.add_argument("--schedule-priority-weight", type=float, default=0.0,
                        help="离线分核时，最早就绪策略兼顾剩余关键路径的权重")
    parser.add_argument("--oproj-n256", action="store_true",
                        help="输出投影复用原N256列分块；精度仍按原exact门槛检查")
    parser.add_argument("--oproj-k512", action="store_true",
                        help="T<=96的O-A按实际行对齐M并用K512预取，N仍128")
    parser.add_argument("--oproj-phase-loop", action="store_true",
                        help="原O-A八个相邻阶段共享循环体，数学/记录/依赖顺序不变")
    parser.add_argument("--hc-stream-pre", action="store_true",
                        help="T96的HC前段采用48 AIV分块供数，保持原点积与RMS累加顺序")
    parser.add_argument("--reuse-gm", action="store_true",
                        help="按完整任务DAG复用等大小临时GM；控制区和原生状态独占")
    parser.add_argument("--qproj-workers", type=int, choices=(20, 24, 32, 64, 128), default=20,
                        help="只重分主Q投影列块，数学Tile和K累加顺序不变")
    parser.add_argument("--qk-vector-indices", action="store_true",
                        help="QK/PV压缩行地址在UB用Vector批量计算，保留原逐行DMA")
    parser.add_argument("--rope-flat-gather", action="store_true",
                        help="三处RoPE逐行gather合并为批量gather，保留原索引与GM访问")
    parser.add_argument("--qk-early-kv", action="store_true",
                        help="T96现选向量索引路径提前通知KV，不改变数学与FFTS事件数量")
    parser.add_argument("--qk-role-split", action="store_true",
                        help="T96 AIV0采KV、AIV1处理64head数学；重新证明packed输出所有权")
    parser.add_argument("--qk-role-wide-heads", action="store_true",
                        help="双AIV分工采用已过CPU规划的32head softmax/16head PV/8head发布")
    parser.add_argument("--qk-query-metadata", action="store_true",
                        help="每query一次准备512个压缩KV地址，四个块复用本核UB元数据")
    parser.add_argument("--qk-all-valid", action="store_true",
                        help="核内先判断当前组所有块均有效，成立时使用去掉重复块判断的循环")
    parser.add_argument("--qk-direct-l1", action="store_true",
                        help="Cube直接读取Native KV到三槽L1；Vector预备地址并确认PV消费")
    parser.add_argument("--row-dependencies", action="store_true",
                        help="T96长Indexer按已证明query/head范围收窄成员依赖，数学与分核保持")
    parser.add_argument("--restore-weight-cache-policy", action="store_true",
                        help="恢复原数学体QA/主Q/KV/O-A只读权重的BYPASS声明，不改变数学或调度")
    parser.add_argument("--indexer-input-dependencies", action="store_true",
                        help="T96进一步按query收窄quant、系数与Indexer之间的输入等待")
    parser.add_argument("--indexer-n128", action="store_true",
                        help="长历史S6 Indexer N128/两组三query共享Key候选")
    parser.add_argument("--indexer-n80", action="store_true",
                        help="长历史B16的M384/N80候选，保留WS K384与原叶计划")
    parser.add_argument("--indexer-n80-ws128", action="store_true",
                        help="N80的WS拆三段K128以恢复下一Key的Right预取")
    parser.add_argument("--indexer-short-tail", action="store_true",
                        help="长历史S6的最后分数块仅剩1..64候选时，只算一个N64面板")
    parser.add_argument("--indexer-panel-group", type=int, choices=(0, 2, 4, 8), default=0,
                        help="N64面板组内展开、组间循环，避免Right Tile跨循环携带")
    parser.add_argument("--indexer-transfer-2048", action="store_true",
                        help="保留原16个N64展开面板，将两次1024分数交接合并为一次2048交接")
    parser.add_argument("--indexer-n32-acc", action="store_true",
                        help="保留M384与WS K384，拆32个N32面板以容纳两份QK累加区")
    parser.add_argument("--indexer-l1-ring", action="store_true",
                        help="N64 score双槽L1循环，Key仅在本轮使用；先审计实际地址再执行")
    parser.add_argument("--indexer-key-pairs", action="store_true",
                        help="N64循环每次两面板，在第一块QK前提交第二块Key的GM到L1")
    parser.add_argument("--indexer-ws-load-order", action="store_true",
                        help="保留首对加载顺序，其余14对先提交QK再加载前块WS Right")
    parser.add_argument("--drop-kv-touch", action="store_true",
                        help="以显式写回到QK依赖替代原kv_touch自拷贝，保留发布和读前维护")
    parser.add_argument("--kv-touch-signal-only", action="store_true",
                        help="保留KV写回汇总通知及等待表，只去掉标记自拷贝和payload屏障")
    parser.add_argument("--ramp-wait", action="store_true",
                        help="DCCI轮询从4tick逐次倍增到wait-backoff-ticks，保持原通知发布")
    parser.add_argument("--hc-mix-resident", action="store_true",
                        help="B16的HC mix整行留UB，保持16×K256 RMS顺序并移除mixed回读")
    parser.add_argument("--hc-mix-chunk-buffer", action="store_true",
                        help="B16保留12成员和原K256流水，用连续BF16块UB缓冲替代norm回读")
    parser.add_argument("--hc-front-fusion", action="store_true",
                        help="T96同一AIV连续执行8行reduce、split及原mix/norm，Sinkhorn保持独立")
    parser.add_argument("--hc-front-tiles", action="store_true",
                        help="HC三阶段融合内部改为Tile直传，保留GM写出和阶段外发布屏障")
    parser.add_argument("--indexer-scale-owners", action="store_true",
                        help="T96的Indexer scale按物理页分配8个写者，保留同页compact写入次序")
    parser.add_argument("--qproj-n128", action="store_true",
                        help="B16主Q投影N128，保留K256及20逻辑worker，尝试实现Right双缓冲")
    parser.add_argument("--qproj-l0-prefetch", action="store_true",
                        help="B16主Q M96/N256/K128显式交错L0预取，QR驻留L1")
    parser.add_argument("--qproj-l1-columns", action="store_true",
                        help="B16主Q双N128列块驻留L1，保持原列分工，交错两个K256累加链")
    parser.add_argument("--paired-qproj-spmd-only", action="store_true",
                        help="仅将主Q双列候选与当前row-deps单SPMD同进程采三窗L1，不重复metadata/Event")
    parser.add_argument("--paired-kv-signal-spmd-only", action="store_true",
                        help="仅同进程比较KV纯信号候选与现选单SPMD：2次预热、3窗L1")
    parser.add_argument("--indexer-leaf16384", action="store_true",
                        help="T96长历史Indexer每叶最多16384候选，保持原分数数学")
    parser.add_argument("--indexer-score-pingpong", action="store_true",
                        help="T96长历史Indexer使用M192/N64双Acc候选，原叶计划不变")
    parser.add_argument("--qk-full-blocks", action="store_true",
                        help="QK稀疏索引全有效时展开8行搬运，否则保留原循环")
    parser.add_argument("--qk-cache-masks", action="store_true",
                        help="QK/PV AIV一次读取本核query的有效块标记到UB")
    parser.add_argument("--qk-mgather", action="store_true",
                        help="QK压缩KV使用现有A3 MGATHER行搬运宏，负索引补零")
    parser.add_argument("--qk-scoped-zero", action="store_true",
                        help="QK保留原逐行DMA，将SWA的零初始化移入分支")
    parser.add_argument("--qk-valid-gather", action="store_true",
                        help="QK全有效压缩块直接覆盖不清零，负索引保留原清零回退")
    parser.add_argument("--qk-chunked-gather", action="store_true",
                        help="每个AIV分两个32行块写原KV交接区，保持原数学/通知")
    parser.add_argument("--qk-merge-heads", action="store_true",
                        help="PV循环状态拆成两个16-head块，保持每元素乘加顺序及原FFTS")
    parser.add_argument("--hc-rows", type=int, choices=(2, 4, 8), default=8,
                        help="HC前端widen/RMS和mix/norm的每个逻辑分片行数")
    parser.add_argument("--indexer-page-views", action="store_true",
                        help="Indexer Key页直接使用device字节偏移视图，去掉奇偶分支")
    parser.add_argument("--specialize-core-identity", action="store_true",
                        help="诊断：为每个物理核身份生成 C++ 模板特化，使调度条件常量折叠")
    parser.add_argument("--precompiled-core-artifact", type=Path,
                        help="复用源码逐字相同、编译上下文一致的 CPU 预编译二进制")
    parser.add_argument("--preserve-all-math", action="store_true",
                        help="诊断：全部原动态数学核按原局部流水调用，外层仅保留控制与信号")
    args = parser.parse_args()
    range_policy = None
    if args.range_policy:
        from range_policy import configure_device_arguments

        range_policy = configure_device_arguments(parser, args)
    if args.hc_compact_stream and not args.range_policy:
        raise ValueError("紧凑HC候选只在正式区间入口上验证，不叠加旧实验分支")
    if args.indexer_ws_load_order and (not all((args.row_dependencies, args.qk_vector_indices,
            args.cached_wait, args.balanced_schedule, args.schedule_cost_profile))
            or any((args.batch != 16, args.history != 131072, args.ramp_wait,
            args.paired_selected_spmd, args.paired_qproj_spmd_only, args.paired_kv_signal_spmd_only,
            args.hc_stream_pre, args.oproj_phase_loop, args.stage_clock,
            args.internal_wait_clock, args.qk_phase_clock, args.indexer_phase_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages,
            args.phase_memory, args.compact_abi_reference, args.specialize_core_identity,
            args.workspace_base, args.pipe_profile))):
        raise ValueError("WS加载顺序先独立验证128K/B16现选策略，不组合探针或阶段替换")
    if (args.indexer_l1_ring or args.indexer_key_pairs) and (not all((args.row_dependencies, args.qk_vector_indices,
            args.cached_wait, args.balanced_schedule, args.schedule_cost_profile))
            or any((args.batch != 16, args.history != 131072, args.ramp_wait,
            args.paired_selected_spmd, args.paired_qproj_spmd_only,
            args.paired_kv_signal_spmd_only, args.hc_stream_pre, args.oproj_phase_loop,
            args.stage_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.preserve_all_math, args.preserve_math_pipeline,
            args.independent_stages, args.phase_memory, args.compact_abi_reference,
            args.specialize_core_identity, args.workspace_base, args.pipe_profile))):
        raise ValueError("N64 score环先独立验证128K/B16现选策略，不组合探针或阶段替换")
    if args.paired_kv_signal_spmd_only and (not args.kv_touch_signal_only or any((
            args.paired_qproj_spmd_only, args.paired_selected_spmd,
            args.metadata_replay_only, args.pipe_profile))):
        raise ValueError("KV纯信号同进程对照需要对应候选，不叠加其他采样模式")
    if (args.drop_kv_touch or args.kv_touch_signal_only) and (not all((args.row_dependencies, args.qk_vector_indices,
            args.cached_wait, args.balanced_schedule, args.schedule_cost_profile))
            or any((args.batch != 16, args.history != 131072, args.ramp_wait,
            args.paired_selected_spmd, args.paired_qproj_spmd_only, args.hc_stream_pre,
            args.oproj_phase_loop, args.stage_clock, args.internal_wait_clock,
            args.qk_phase_clock, args.indexer_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.phase_memory,
            args.compact_abi_reference, args.specialize_core_identity, args.workspace_base))):
        raise ValueError("KV标记折叠先用128K/B16现选策略，不组合其他探针或阶段替换")
    if args.indexer_n32_acc and any((args.batch != 16, args.history != 131072,
            args.ramp_wait, args.paired_selected_spmd, args.paired_qproj_spmd_only,
            args.hc_stream_pre, args.oproj_phase_loop, args.stage_clock,
            args.internal_wait_clock, args.qk_phase_clock, args.indexer_phase_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages,
            args.phase_memory, args.compact_abi_reference, args.specialize_core_identity,
            args.workspace_base)):
        raise ValueError("N32双Acc先独立验证128K/B16原控制，不组合阶段替换或探针")
    if args.paired_qproj_spmd_only and (not all((args.qproj_l1_columns,
            args.row_dependencies, args.cached_wait, args.balanced_schedule,
            args.schedule_cost_profile, args.qk_vector_indices))
            or args.metadata_replay_only or args.paired_selected_spmd or args.pipe_profile):
        raise ValueError("主Q同进程对照需现选row-deps/QK地址缓存/退避策略，且不组合其他采样")
    if args.qproj_l1_columns and any((args.batch != 16, args.history != 131072,
            args.ramp_wait, args.paired_selected_spmd, args.hc_stream_pre,
            args.oproj_phase_loop, args.stage_clock, args.internal_wait_clock,
            args.qk_phase_clock, args.indexer_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.phase_memory,
            args.compact_abi_reference, args.specialize_core_identity, args.workspace_base)):
        raise ValueError("主Q双列驻L1先独立验证128K/B16，不组合阶段替换或探针")
    if args.indexer_transfer_2048 and any((args.batch != 16, args.history != 131072,
            args.ramp_wait, args.indexer_panel_group, args.indexer_short_tail,
            args.paired_selected_spmd, args.hc_stream_pre, args.oproj_phase_loop,
            args.stage_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.preserve_all_math, args.preserve_math_pipeline,
            args.independent_stages, args.phase_memory, args.compact_abi_reference,
            args.specialize_core_identity, args.workspace_base)):
        raise ValueError("2048交接先独立验证128K/B16原控制，不组合其他新候选或探针")
    if args.indexer_panel_group and any((args.batch != 16, args.ramp_wait,
            args.paired_selected_spmd, args.hc_stream_pre, args.oproj_phase_loop,
            args.stage_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.preserve_all_math, args.preserve_math_pipeline,
            args.independent_stages, args.phase_memory, args.compact_abi_reference,
            args.specialize_core_identity, args.workspace_base)):
        raise ValueError("N64分组循环先只验证B16原控制，不组合读侧或阶段替换/探针")
    if args.indexer_short_tail and any((args.batch not in (16, 24), args.paired_selected_spmd,
            args.hc_stream_pre, args.oproj_phase_loop, args.stage_clock, args.internal_wait_clock,
            args.qk_phase_clock, args.indexer_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.phase_memory,
            args.compact_abi_reference, args.specialize_core_identity, args.workspace_base)):
        raise ValueError("短尾先验证B16/B24原控制，不组合阶段替换或探针")
    if args.ramp_wait and (not args.cached_wait or args.wait_backoff_ticks < 4 or any((
            args.batch != 16, args.indexer_short_tail, args.paired_selected_spmd,
            args.hc_stream_pre, args.oproj_phase_loop, args.stage_clock, args.internal_wait_clock,
            args.qk_phase_clock, args.indexer_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.phase_memory,
            args.compact_abi_reference, args.specialize_core_identity, args.workspace_base))):
        raise ValueError("递增退避先独立验证B16 DCCI读侧，不与数学变体/探针/阶段替换组合")
    if args.indexer_scale_owners and any((args.batch != 16, args.paired_selected_spmd,
            args.hc_stream_pre, args.oproj_phase_loop, args.stage_clock, args.internal_wait_clock,
            args.qk_phase_clock, args.indexer_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.phase_memory,
            args.compact_abi_reference, args.specialize_core_identity, args.workspace_base)):
        raise ValueError("scale页分工先验证B16原控制，不组合阶段/代码替换")
    if args.hc_front_tiles and not args.hc_front_fusion:
        raise ValueError("HC Tile直传需同时传--hc-front-fusion")
    if args.hc_front_fusion and any((args.batch != 16, args.hc_rows != 8,
            args.hc_mix_resident, args.hc_stream_pre, args.paired_selected_spmd,
            args.stage_clock, args.internal_wait_clock, args.qk_phase_clock, args.indexer_phase_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages,
            args.phase_memory, args.compact_abi_reference, args.specialize_core_identity,
            args.workspace_base)):
        raise ValueError("HC三阶段融合先验证B16原控制，不叠加其他HC/阶段替换或探针")
    if args.qproj_l0_prefetch and any((args.batch != 16, args.qproj_workers != 20,
            args.qproj_n128, args.oproj_k512, args.oproj_n256, args.paired_selected_spmd,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("主Q显式L0候选先用B16原20worker，不组合其他主Q/逻辑IR改写")
    if args.qproj_n128 and any((args.batch != 16, args.qproj_workers != 20,
            args.oproj_k512, args.oproj_n256, args.paired_selected_spmd,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("主Q N128先在B16完整原控制中独立核对")
    if args.hc_mix_resident and any((args.batch != 16, args.hc_rows != 8,
            args.hc_stream_pre, args.paired_selected_spmd, args.stage_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("整行HC mix先用B16完整原控制，不能叠加HC或阶段替换")
    if args.indexer_n80 and any((args.batch != 16, args.indexer_n128, args.indexer_page_views,
            args.indexer_leaf16384, args.indexer_score_pingpong, args.paired_selected_spmd,
            args.stage_clock, args.internal_wait_clock, args.indexer_phase_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("N80先检查B16完整原控制路径，不与其他Indexer候选或探针叠加")
    if args.metadata_history and not args.metadata_replay_only:
        raise ValueError("显式metadata历史仅用于metadata重放任务")
    if args.paired_selected_spmd and not (args.metadata_replay_only and args.row_dependencies
            and args.cached_wait and args.balanced_schedule and args.schedule_cost_profile):
        raise ValueError("同进程旧/新对比仅用于行依赖候选的metadata功能任务")
    if args.row_dependencies and any((args.batch != 16, args.hc_stream_pre,
            args.oproj_phase_loop, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.preserve_all_math, args.preserve_math_pipeline,
            (args.independent_stages and not args.cached_wait), args.phase_memory, args.compact_abi_reference,
            args.specialize_core_identity, args.workspace_base)):
        raise ValueError("行依赖候选仅B16原控制与数学；先不组合阶段变换或内部探针")
    if args.indexer_input_dependencies and (not args.row_dependencies or args.paired_selected_spmd):
        raise ValueError("Indexer输入行依赖要求已有行依赖，先不混用旧版配对metadata任务")
    if args.hc_mix_chunk_buffer and any((not args.row_dependencies, args.paired_selected_spmd,
                                       args.stage_clock)):
        raise ValueError("HC分块UB候选先保持现选原控制，不叠加配对metadata或阶段插桩")
    if args.qk_direct_l1 and any((args.batch != 16, args.qk_merge_heads, args.qk_chunked_gather,
            args.qk_mgather, args.qk_valid_gather, args.qk_scoped_zero, args.qk_full_blocks,
            args.qk_cache_masks, args.hc_stream_pre, args.indexer_score_pingpong, args.indexer_leaf16384,
            args.internal_wait_clock, args.qk_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages, args.reuse_gm,
            args.precise_payload_cache, args.phase_memory, args.compact_abi_reference)):
        raise ValueError("直接L1只组合B16/S6原逐核控制和独占临时区，不叠加其他数学候选")
    if args.qk_merge_heads and any((args.batch != 16, args.qk_chunked_gather,
            args.qk_mgather, args.qk_valid_gather, args.qk_scoped_zero, args.qk_full_blocks,
            args.hc_stream_pre, args.indexer_score_pingpong, args.indexer_leaf16384,
            args.internal_wait_clock, args.qk_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("PV分head合并仅组合B16/S6原控制，不叠加其他数学候选")
    if args.indexer_score_pingpong and any((args.batch != 16, args.indexer_n128,
            args.indexer_leaf16384, args.indexer_page_views, args.hc_stream_pre,
            args.indexer_phase_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.preserve_all_math, args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("双Acc候选只组合T96原控制，不复用原Indexer内部循环探针")
    if args.indexer_leaf16384 and any((args.batch != 16, args.indexer_n128,
            args.indexer_page_views, args.hc_stream_pre, args.indexer_phase_clock,
            args.internal_wait_clock, args.qk_phase_clock, args.preserve_all_math,
            args.preserve_math_pipeline, args.independent_stages)):
        raise ValueError("16384叶候选只组合T96原分数数学，不复用原内部循环探针")
    if args.hc_stream_pre and any((args.batch != 16, args.hc_rows != 8,
            args.stage_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.compile_schedule, args.precise_payload_cache,
            args.task_completion, args.independent_stages, args.phase_memory,
            args.preserve_math_pipeline, args.preserve_all_math, args.workspace_base,
            args.compact_abi_reference, args.specialize_core_identity, args.aggregate_completion,
            args.reuse_gm, args.oproj_phase_loop, args.qk_chunked_gather)):
        raise ValueError("HC流式候选仅在T96、原逐核控制和独占工作区上验证，不组合其他阶段变换")
    if args.oproj_phase_loop and any((args.stage_clock, args.internal_wait_clock, args.qk_phase_clock,
            args.indexer_phase_clock, args.compile_schedule, args.precise_payload_cache,
            args.task_completion, args.independent_stages, args.phase_memory,
            args.preserve_math_pipeline, args.preserve_all_math, args.workspace_base,
            args.compact_abi_reference, args.specialize_core_identity, args.aggregate_completion,
            args.oproj_n256, args.oproj_k512)):
        raise ValueError("O-A阶段循环候选仅组合原N128数学/逐核信号，不能复用原阶段探针位置")
    if args.metadata_replay_only and any((args.stage_clock, args.internal_wait_clock,
                                         args.qk_phase_clock, args.indexer_phase_clock, args.pipe_profile)):
        raise ValueError("同地址metadata功能检查不与性能探针混用")
    if sum((args.qk_phase_clock, args.internal_wait_clock, args.indexer_phase_clock)) > 1:
        raise ValueError("两种局部探针不能共用进度空闲区")
    if args.indexer_phase_clock and (args.indexer_n128 or args.indexer_page_views):
        raise ValueError("Indexer区间探针只支持原N64和原页读取体")
    if (args.internal_wait_clock or args.qk_phase_clock or args.indexer_phase_clock) and any((args.stage_clock, args.compile_schedule,
            args.precise_payload_cache, args.task_completion, args.independent_stages,
            args.phase_memory, args.preserve_math_pipeline, args.preserve_all_math,
            args.workspace_base, args.compact_abi_reference, args.specialize_core_identity,
            args.aggregate_completion)):
        raise ValueError("内部等待探针仅使用原逐核信号与默认代码生成，不与其他探针共用进度空闲区")
    if args.cached_wait and (args.bypass_wait or args.task_completion
                            or args.compact_abi_reference or args.specialize_core_identity
                            or args.preserve_all_math):
        raise ValueError("cached 等待诊断本轮只与原始分核数学体组合")
    if args.cached_wait and args.independent_stages and not (
            args.stage_kernels == "*" and args.compact_stage_inputs and args.stage_noinline
            and args.batch == 16 and args.history == 131072 and args.qk_vector_indices
            and args.row_dependencies and args.balanced_schedule
            and args.schedule_policy == "earliest_ready" and args.wait_backoff_ticks == 100
            and not args.ramp_wait and not args.rope_flat_gather):
        raise ValueError("cached与独立阶段仅开放CPU已核对的B16全阶段noinline/100tick组合；不允许部分阶段混用")
    if args.preserve_all_math and any((args.independent_stages,
            args.phase_memory, args.compile_schedule, args.precise_payload_cache,
            args.specialize_core_identity, args.workspace_base, args.compact_abi_reference,
            args.bypass_wait, args.task_completion, args.preserve_math_pipeline)):
        raise ValueError("全部原数学核组合先单独检查，不与其他生成变换混用")
    if args.precompiled_core_artifact and not (args.specialize_core_identity or args.preserve_all_math):
        raise ValueError("此 CPU 预编译复用目前仅用于核身份/全部数学核候选")
    if args.specialize_core_identity and (not args.compile_schedule or any((args.stage_clock,
            args.independent_stages, args.phase_memory, args.preserve_math_pipeline,
            args.workspace_base, args.compact_abi_reference, args.bypass_wait, args.task_completion))):
        raise ValueError("核身份特化要求 --compile-schedule，且不与其他代码生成诊断混用")
    if args.schedule_cost_profile and not args.balanced_schedule:
        raise ValueError("实测阶段成本要求显式启用 --balanced-schedule")
    if args.compact_abi_reference and any((args.stage_clock, args.compile_schedule,
            args.precise_payload_cache, args.independent_stages, args.phase_memory,
            args.preserve_math_pipeline, args.workspace_base, args.bypass_wait, args.task_completion)):
        raise ValueError("实际 ABI 裁剪诊断不与其他编译变换组合")
    if args.stage_clock and (args.compile_schedule or args.precise_payload_cache):
        raise ValueError("旧阶段探针尚不支持该控制流形状；不可误用时戳边界")
    if args.independent_stages and (args.stage_clock or args.phase_memory or args.compile_schedule
                                    or args.precise_payload_cache or args.preserve_math_pipeline):
        raise ValueError("独立阶段归因不与其他编译/边界变换组合")
    if args.wait_backoff_ticks and not (args.bypass_wait or args.cached_wait):
        raise ValueError("轮询间隔实验要求显式启用 --bypass-wait 或 --cached-wait")
    if args.task_completion and (args.stage_clock or args.independent_stages or args.phase_memory
                                 or args.preserve_math_pipeline or args.workspace_base):
        raise ValueError("任务计数候选不与其他编译/插桩变换组合")
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须通过统一队列分配单卡")
    batch, history = args.batch, args.history
    tokens = batch * 6
    if not 1 <= batch <= 40 or history < 0:
        raise ValueError("当前测试 fixture 要求 1<=B<=40，非负 history")
    report = {"passed": False, "stage": "initializing", "history": history, "batch": batch,
              "implementation_kind": "historical_shape_specialized_spmd",
              "target_dynamic_shape": False,
              "range_policy": range_policy.report() if range_policy else None,
              "query_tokens": 6, "tokens": tokens, "device": args.device, "native_forward_executed": False,
              "scope": "正式第4层权重，完整CSA attention半层，合成历史；固定输入图重放",
              "metadata_replay_only": args.metadata_replay_only,
              "indexer_leaf16384": args.indexer_leaf16384,
              "indexer_score_pingpong": args.indexer_score_pingpong,
              "qk_merge_heads": args.qk_merge_heads,
              "qk_direct_l1": args.qk_direct_l1,
              "row_dependencies": args.row_dependencies,
              "restore_weight_cache_policy": args.restore_weight_cache_policy,
              "indexer_input_dependencies": args.indexer_input_dependencies,
              "hc_stream_pre_enabled": args.hc_stream_pre,
              "hc_compact_stream_enabled": args.hc_compact_stream,
              "fixture_restored_per_call": True, "signal_reset_between_calls": False,
              "warmup": args.warmup, "iterations": args.iters, "l1_windows_per_side": args.windows,
              "consecutive_decode_trajectory_validated": False, "l1_windows": [],
              "stage_clock_enabled": args.stage_clock, "bypass_wait": args.bypass_wait,
              "internal_wait_clock_enabled": args.internal_wait_clock,
              "qk_phase_clock_enabled": args.qk_phase_clock,
              "indexer_phase_clock_enabled": args.indexer_phase_clock,
              "cached_wait": args.cached_wait,
              "wait_backoff_ticks": args.wait_backoff_ticks,
              "ramp_wait": args.ramp_wait,
              "aggregate_completion": args.aggregate_completion,
              "task_completion": args.task_completion,
              "pipe_profile": args.pipe_profile,
              "balanced_schedule_enabled": args.balanced_schedule,
              "phase_memory_enabled": args.phase_memory,
              "compiled_schedule_enabled": args.compile_schedule,
              "precise_payload_cache_enabled": args.precise_payload_cache,
              "schedule_policy": args.schedule_policy,
              "independent_stage_codegen": args.independent_stages,
              "independent_stage_selection": args.stage_kernels,
              "compact_stage_inputs": args.compact_stage_inputs,
              "independent_stage_noinline": args.stage_noinline,
              "workspace_base_codegen": args.workspace_base,
              "compact_abi_reference": str(args.compact_abi_reference) if args.compact_abi_reference else None,
              "schedule_cost_profile": str(args.schedule_cost_profile) if args.schedule_cost_profile else None,
              "rotate_wait_order": args.rotate_wait_order,
              "schedule_priority_weight": args.schedule_priority_weight,
              "oproj_n_tile": 256 if args.oproj_n256 else 128,
              "oproj_k512": args.oproj_k512,
              "oproj_phase_loop_enabled": args.oproj_phase_loop,
              "gm_reuse_enabled": args.reuse_gm,
              "qproj_workers": args.qproj_workers,
              "qk_vector_indices": args.qk_vector_indices,
              "rope_flat_gather": args.rope_flat_gather,
              "qk_query_metadata": args.qk_query_metadata,
              "qk_early_kv": args.qk_early_kv,
              "qk_role_split": args.qk_role_split,
              "qk_role_wide_heads": args.qk_role_wide_heads,
              "qk_all_valid": args.qk_all_valid,
              "indexer_n128": args.indexer_n128,
              "indexer_n80": args.indexer_n80,
              "indexer_n80_ws128": args.indexer_n80_ws128,
              "indexer_short_tail": args.indexer_short_tail,
              "indexer_panel_group": args.indexer_panel_group,
              "indexer_transfer_2048": args.indexer_transfer_2048,
              "indexer_n32_acc": args.indexer_n32_acc,
              "indexer_l1_ring": args.indexer_l1_ring,
              "indexer_key_pairs": args.indexer_key_pairs,
              "indexer_ws_load_order": args.indexer_ws_load_order,
              "drop_kv_touch": args.drop_kv_touch,
              "kv_touch_signal_only": args.kv_touch_signal_only,
              "hc_mix_resident": args.hc_mix_resident,
              "hc_front_fusion": args.hc_front_fusion,
              "hc_mix_chunk_buffer": args.hc_mix_chunk_buffer,
              "hc_front_tiles": args.hc_front_tiles,
              "indexer_scale_owners": args.indexer_scale_owners,
              "qproj_n128": args.qproj_n128,
              "qproj_l0_prefetch": args.qproj_l0_prefetch,
              "qproj_l1_columns": args.qproj_l1_columns,
              "paired_qproj_spmd_only": args.paired_qproj_spmd_only,
              "paired_kv_signal_spmd_only": args.paired_kv_signal_spmd_only,
              "qk_full_blocks": args.qk_full_blocks,
              "qk_cache_masks": args.qk_cache_masks,
              "qk_mgather": args.qk_mgather,
              "qk_scoped_zero": args.qk_scoped_zero,
              "qk_valid_gather": args.qk_valid_gather,
              "qk_chunked_gather": args.qk_chunked_gather,
              "hc_rows": args.hc_rows,
              "indexer_page_views": args.indexer_page_views,
              "core_identity_specialization": args.specialize_core_identity,
              "preserve_all_math": args.preserve_all_math,
              "precompiled_core_artifact": str(args.precompiled_core_artifact) if args.precompiled_core_artifact else None,
              "preserved_math_pipeline": args.preserve_math_pipeline,
              "eligible_for_700us_target": not (args.metadata_replay_only or args.stage_clock or args.internal_wait_clock or args.qk_phase_clock or args.indexer_phase_clock or args.preserve_math_pipeline
                                               or args.independent_stages or args.workspace_base
                                               or args.compact_abi_reference or args.specialize_core_identity
                                               or args.preserve_all_math)}
    # 兼容旧结果字段；新的验收使用同轮相对耗时，700 μs 已不再是目标。
    report["eligible_for_performance_target"] = report["eligible_for_700us_target"]

    def stage(value):
        report["stage"] = value
        save(args.output / "result.json", report)
        print(value, flush=True)

    try:
        report["environment"] = verify_isolation()
        import vllm_ascend.ops.pypto as operator_package
        if args.range_policy:
            # 基线随后只搜索冻结源码；先显式加载正式 SPMD 的独立信号模块，
            # 避免生成入口的绝对 import 被基线快照遮蔽，不增加基线搜索目录。
            signal_module = importlib.import_module(
                "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.signals")
            if Path(signal_module.__file__).resolve() != SIGNAL_SOURCE.resolve():
                raise RuntimeError("正式 SPMD 信号模块来自非当前工作区")
            report["production_signal_source"] = str(SIGNAL_SOURCE.resolve())
        frozen = Path(os.environ["HC_PRE_TEST_OPERATOR_SOURCE"]).resolve()
        operator_package.__path__ = [str(frozen)]
        report["frozen_operator_source"] = str(frozen)
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op
        if not enable_custom_op():
            raise RuntimeError("fixture 初始化所需自定义算子未加载")
        additional = decode_additional_config(SimpleNamespace(
            graph_mode="full_decode_only", capture_sizes=[tokens], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False,
                          pto_csa_ring_config={"heap_mb": [256, 128, 256, 32], "task_window": 4096})
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [tokens],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs, variant_package
        adapter = importlib.import_module(f"{variant_package()}.native_adapter")
        assert frozen in Path(adapter.__file__).resolve().parents
        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, batch, history, 1024, device)
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            stage("formal_weights_and_fixture_ready")
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             enable_chip_swimlane=1, enable_dep_gen=True,
                             output_dir=str(args.output / "dfx"), **ring_sizing_kwargs(additional))
            if args.qk_mgather:
                from qk_mgather_boundary import validate

                report["qk_mgather_boundary"] = validate(
                    device, CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO))
                stage("qk_mgather_boundaries_passed")
            groups = {name: (fixture["metadata"][group["prefix"]], group["views"])
                      for name, group in fixture["groups"].items()}
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            call = adapter.NativeCSACall(
                adapter.CSAOperators.register(), weights, fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name, compact_metadata=fixture["compact"],
                buffers={"idx_topk_scores": torch.empty((tokens, 512), dtype=torch.float32, device=device),
                         "idx_topk": torch.empty((tokens, 512), dtype=torch.int32, device=device), "x_out": output})
            stage("binding_actual_descriptors")
            descriptors = descriptors_from_tensors(call.args)
            if range_policy is not None:
                from range_policy import exact_cost_profile, policy_from_descriptors

                actual_policy = policy_from_descriptors(descriptors, max_seq_len=history + 6)
                if actual_policy != range_policy:
                    raise ValueError("真实描述符与入口的工作量区间不一致")
                args.schedule_cost_profile = exact_cost_profile(descriptors, range_policy)
                report["schedule_cost_profile"] = (
                    str(args.schedule_cost_profile) if args.schedule_cost_profile else None)
                report["range_policy"]["cost_profile"] = (
                    str(args.schedule_cost_profile) if args.schedule_cost_profile else None)
                report["range_policy"]["cost_is_same_shape_measurement"] = bool(args.schedule_cost_profile)
            if range_policy is not None:
                from range_policy import build_range_variant

                args.case, preparation = build_range_variant(
                    descriptors, max_seq_len=history + 6,
                    output_root=args.output / "specialization", case_name=f"h{history}_b{batch}")
            else:
                args.case, preparation = build_variant(
                    descriptors, max_seq_len=history + 6,
                    output_root=args.output / "specialization", case_name=f"h{history}_b{batch}",
                    aggregate_completion=args.aggregate_completion,
                    balanced_schedule=args.balanced_schedule, compile_schedule=args.compile_schedule,
                    precise_payload_cache=args.precise_payload_cache, schedule_policy=args.schedule_policy,
                    task_completion=args.task_completion, cost_profile=args.schedule_cost_profile,
                    rotate_wait_order=args.rotate_wait_order, priority_weight=args.schedule_priority_weight,
                    oproj_n_tile=256 if args.oproj_n256 else 128, reuse_gm=args.reuse_gm,
                    qproj_workers=args.qproj_workers, qk_vector_indices=args.qk_vector_indices,
                    indexer_n128=args.indexer_n128, qk_full_blocks=args.qk_full_blocks,
                    qk_cache_masks=args.qk_cache_masks, qk_mgather=args.qk_mgather,
                    hc_rows=args.hc_rows, indexer_page_views=args.indexer_page_views,
                    qk_scoped_zero=args.qk_scoped_zero, qk_valid_gather=args.qk_valid_gather,
                    oproj_k512=args.oproj_k512, qk_chunked_gather=args.qk_chunked_gather,
                    indexer_leaf16384=args.indexer_leaf16384,
                    indexer_score_pingpong=args.indexer_score_pingpong,
                    qk_merge_heads=args.qk_merge_heads, qk_direct_l1=args.qk_direct_l1,
                    row_dependencies=args.row_dependencies, indexer_n80=args.indexer_n80,
                    hc_mix_resident=args.hc_mix_resident, qproj_n128=args.qproj_n128,
                    qproj_l0_prefetch=args.qproj_l0_prefetch, hc_front_fusion=args.hc_front_fusion,
                    qproj_l1_columns=args.qproj_l1_columns,
                    hc_front_tiles=args.hc_front_tiles, indexer_scale_owners=args.indexer_scale_owners,
                    indexer_input_dependencies=args.indexer_input_dependencies,
                    hc_mix_chunk_buffer=args.hc_mix_chunk_buffer,
                    restore_weight_cache_policy=args.restore_weight_cache_policy,
                    qk_query_metadata=args.qk_query_metadata,
                    indexer_n80_ws128=args.indexer_n80_ws128,
                    qk_all_valid=args.qk_all_valid, indexer_short_tail=args.indexer_short_tail,
                    indexer_panel_group=args.indexer_panel_group,
                    indexer_transfer_2048=args.indexer_transfer_2048,
                    indexer_n32_acc=args.indexer_n32_acc, drop_kv_touch=args.drop_kv_touch,
                    kv_touch_signal_only=args.kv_touch_signal_only,
                    indexer_l1_ring=args.indexer_l1_ring,
                    indexer_ws_load_order=args.indexer_ws_load_order,
                    indexer_key_pairs=args.indexer_key_pairs, qk_early_kv=args.qk_early_kv,
                    qk_role_split=args.qk_role_split, qk_role_wide_heads=args.qk_role_wide_heads,
                    rope_flat_gather=args.rope_flat_gather,
                    allow_source_schedule=args.range_policy)
            if range_policy is not None:
                preparation["range_policy"] = report["range_policy"]
                save(args.case / "preparation.json", preparation)
            report["shape_specialization"] = {"case": str(args.case), "anchor": preparation["anchor"],
                                             "wait_plan": preparation["wait_plan"],
                                             "completion_aggregation": preparation["completion_aggregation"]}
            report["shape_specialization"]["balanced_schedule"] = preparation["balanced_schedule"]
            report["shape_specialization"]["gm_reuse"] = preparation["gm_reuse"]
            layout = json.loads((args.case / "layout.json").read_text())
            if args.hc_compact_stream:
                from hc_pre_direct_stream_v1.transform import transform

                prior_source = (args.case / "entry_jit.py").read_text()
                prior_bindings = json.loads((args.case / "entry.bindings.json").read_text())
                source, bindings, audit = transform(prior_source, layout, prior_bindings)
                (args.case / "entry_jit_before_hc_compact.py").write_text(prior_source)
                save(args.case / "entry.bindings.before_hc_compact.json", prior_bindings)
                (args.case / "entry_jit.py").write_text(source)
                save(args.case / "entry.bindings.json", bindings)
                save(args.output / "hc_compact_stream.json", audit)
                shutil.copy2(Path(__file__).parent / "hc_pre_direct_stream_v1/transform.py",
                             args.output / "hc_compact_stream_transform.py")
                report["hc_compact_stream"] = audit
                # 原CPU进度模型只证明变换前的跨task等待；FIFO取代的边单独验证。
                preparation["baseline_cpu_protocol"] = preparation.pop("cpu_protocol")
                preparation["hc_compact_stream"] = audit
                save(args.case / "preparation.json", preparation)
            progress_expectation(layout, 2 + args.warmup + args.iters + args.windows)
            if range_policy is not None:
                from historical_sources import load_historical_module

                generated = SimpleNamespace(
                    bindings=json.loads((args.case / "entry.bindings.json").read_text()),
                    layout=layout,
                    schedule=json.loads((args.case / "schedule.json").read_text()))
                prepared = load_historical_module("arguments").PreparedCSAArguments(
                    generated, call.args, workspace_guard_bytes=512)
            else:
                prepared = PreparedStaticArguments(args.case, call.args, workspace_guard_bytes=512)
            prepared.require_core_capacity(256)
            for name in ("entry_jit.py", "layout.json", "entry.bindings.json"):
                shutil.copy2(args.case / name, args.output / name)
            shutil.copy2(SIGNAL_SOURCE, args.output / "local_signal_adapter.py")
            entry_source, entry_args = args.output / "entry_jit.py", prepared.core_args
            if args.oproj_phase_loop:
                from oproj_phase_loop import deduplicate

                deduplicated, report["oproj_phase_loop"] = deduplicate(entry_source.read_text(), layout)
                entry_source.write_text(deduplicated)
                save(args.output / "oproj_phase_loop.json", report["oproj_phase_loop"])
            if args.hc_stream_pre:
                from hc_stream_pre import transform

                streamed, report["hc_stream_pre"] = transform(entry_source.read_text(), layout)
                entry_source.write_text(streamed)
                save(args.output / "hc_stream_pre.json", report["hc_stream_pre"])
                shutil.copy2(Path(__file__).with_name("hc_stream_pre.py"),
                             args.output / "hc_stream_transform.py")
            if args.compact_abi_reference:
                entry_source, entry_args, report["compact_abi"] = build_compact_entry(
                    args.case, prepared, args.compact_abi_reference, args.output / "compact_abi")
            spec = importlib.util.spec_from_file_location("csa_static_perf_entry", entry_source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            if args.qk_chunked_gather:
                from qk_chunked_boundary import validate

                bundle = next((args.output / "specialization" / "math_variants").rglob(
                    "core_bodies_qk_chunked_gather.py"))
                report["qk_chunked_boundary"] = validate(
                    device, options, bundle, args.output / "qk_chunked_boundary")
                stage("qk_chunked_boundaries_passed")
            if args.qk_valid_gather:
                from qk_valid_gather_boundary import validate

                bundle = next((args.output / "specialization" / "math_variants").rglob(
                    "core_bodies_qk_valid_gather.py"))
                report["qk_valid_gather_boundary"] = validate(
                    device, options, bundle, args.output / "qk_valid_boundary")
                stage("qk_valid_gather_boundaries_passed")
            entries = {
                "dynamic_pto": lambda: adapter.decode_csa_tp1_layer_test(*call.core_args, config=options),
                "single_spmd": lambda: module.entry(*entry_args, config=options),
            }
            initial = {name: group["initial"].to(device) for name, group in fixture["groups"].items()}

            def reset():
                # 恢复在计时及 DFX 窗口外；两边共用权重、metadata、输入和输出地址。
                for name, group in fixture["groups"].items():
                    group["allocation"].copy_(initial[name])
                output.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)

            def check(reference):
                actual = collect_state(fixture, output, call.args["idx_topk"])
                checks = exact_checks(actual, reference)
                if not checks_passed(checks):
                    raise AssertionError("性能路径输出/状态与动态 eager 参考不一致：" + str(checks))
                return checks

            stream = torch.npu.Stream(device=device)
            stream.wait_stream(torch.npu.current_stream())
            graphs, generations = {}, 0
            with torch.npu.stream(stream):
                reset()
                entries["dynamic_pto"]()
                stream.synchronize()
                reference = collect_state(fixture, output, call.args["idx_topk"])
                reset()
                stage("single_spmd_first_compile_and_launch")
                with contextlib.ExitStack() as extensions:
                    if args.precompiled_core_artifact:
                        extensions.enter_context(reuse_identical_core_binaries(
                            args.precompiled_core_artifact, args.output / "precompiled_core_reuse"))
                    if args.specialize_core_identity:
                        extensions.enter_context(specialize_core_identity(
                            prepared.contract["leaf_tensor_count"], args.output / "core_specialization"))
                    if args.task_completion:
                        extensions.enter_context(task_counter_codegen(
                            args.output / "task_completion", layout["task_completion"]["phases"]))
                    if args.workspace_base:
                        extensions.enter_context(workspace_base_codegen(
                            args.case, prepared, args.output / "workspace_base"))
                    if args.independent_stages:
                        extensions.enter_context(independent_stage_codegen(
                            layout, args.output / "independent_stages",
                            selected=tuple(args.stage_kernels.split(",")), compact_params=args.compact_stage_inputs,
                            noinline=args.stage_noinline))
                    if args.phase_memory:
                        extensions.enter_context(scoped_memory_planning(args.output / "phase_memory"))
                    if args.preserve_math_pipeline or args.preserve_all_math:
                        builds = list(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("_decode_csa_tp1_layer_*"))
                        if len(builds) != 1:
                            raise ValueError("本轮动态 PTO 编译目录不唯一")
                        extensions.enter_context(preserved_pipeline(
                            args.case, builds[0], args.output / "preserved_math", all_stages=args.preserve_all_math))
                    if args.bypass_wait or args.cached_wait:
                        extensions.enter_context(bypass_signal_wait(
                            args.output / "signal_polling", args.wait_backoff_ticks,
                            cached=args.cached_wait, ramp_backoff=args.ramp_wait))
                    if args.stage_clock:
                        extensions.enter_context(compiler_probe(
                            layout, prepared.contract["leaf_tensor_count"], args.output / "stage_clock"))
                    if args.internal_wait_clock:
                        from internal_wait_clock import compiler_probe as internal_probe

                        extensions.enter_context(internal_probe(
                            layout, prepared.contract["leaf_tensor_count"], args.output / "internal_wait_clock"))
                    if args.qk_phase_clock:
                        from qk_phase_clock import compiler_probe as phase_probe

                        extensions.enter_context(phase_probe(
                            layout, prepared.contract["leaf_tensor_count"], args.output / "qk_phase_clock"))
                    if args.indexer_phase_clock:
                        from indexer_phase_clock import compiler_probe as indexer_probe

                        extensions.enter_context(indexer_probe(
                            layout, prepared.contract, args.output / "indexer_phase_clock"))
                    if args.cached_wait and args.independent_stages:
                        from outlined_cached_audit import audit as audit_outlined_cached

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result = audit_outlined_cached(
                            artifact.output_dir, args.output / "independent_stages", args.output / "signal_polling")
                        save(args.output / "outlined_cached_codegen_audit.json", audit_result)
                        stage("outlined_cached_codegen_audit_passed_before_execution")
                    if args.indexer_ws_load_order:
                        from indexer_ws_load_order_codegen import inspect as inspect_ws_load_order

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result, indexer_cpp = inspect_ws_load_order(Path(artifact.output_dir) / "ptoas")
                        order = [p["source_order"] for p in audit_result["cpp_issue_order"]]
                        if (not audit_result["all_right_pairs_disjoint"]
                                or not audit_result["all_acc_pairs_disjoint"]
                                or audit_result["directly_serialized_pair_count"]
                                or order != ["load_then_qk"] + ["qk_then_load"] * 14):
                            raise ValueError("WS加载顺序候选实际地址/提交与预期不符，禁止执行")
                        save(args.output / "indexer_ws_order_codegen_audit.json", audit_result)
                        (args.output / "indexer_ws_order_stage.cpp.txt").write_text(indexer_cpp + "\n")
                        stage("indexer_ws_order_codegen_audit_passed_before_execution")
                    if args.qk_role_split:
                        from qk_role_split_codegen_audit import audit as audit_qk_roles

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result = audit_qk_roles(Path(artifact.output_dir) / "ptoas")
                        actual_tiles = tuple(audit_result[key] for key in
                                             ("head_tile", "softmax_tile", "publication_tile"))
                        expected_tiles = (16, 32, 8) if args.qk_role_wide_heads else (8, 8, 8)
                        if actual_tiles != expected_tiles:
                            raise ValueError(f"实际角色分工Tile与请求不符：{actual_tiles}")
                        save(args.output / "qk_role_split_codegen_audit.json", audit_result)
                        stage("qk_role_split_codegen_audit_passed_before_execution")
                    if args.rope_flat_gather:
                        from rope_flat_gather_codegen_audit import audit as audit_rope_gather

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result = audit_rope_gather(Path(artifact.output_dir) / "ptoas")
                        save(args.output / "rope_flat_gather_codegen_audit.json", audit_result)
                        stage("rope_flat_gather_codegen_audit_passed_before_execution")
                    if args.qk_early_kv:
                        from qk_early_kv_codegen_audit import audit as audit_qk_early_kv

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result = audit_qk_early_kv(Path(artifact.output_dir) / "ptoas")
                        save(args.output / "qk_early_kv_codegen_audit.json", audit_result)
                        stage("qk_early_kv_codegen_audit_passed_before_execution")
                    if args.qk_merge_heads:
                        from qk_merge_heads_codegen_audit import audit as audit_qk_merge_heads

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result, merge_cpp = audit_qk_merge_heads(Path(artifact.output_dir) / "ptoas")
                        save(args.output / "qk_merge_heads_codegen_audit.json", audit_result)
                        (args.output / "qk_merge_heads_merge.cpp.txt").write_text(merge_cpp + "\n")
                        stage("qk_merge_heads_codegen_audit_passed_before_execution")
                    if args.indexer_l1_ring or args.indexer_key_pairs:
                        from indexer_l1_ring_codegen_audit import audit as audit_indexer_l1_ring

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        audit_result, stage_cpp, loop_cpp = audit_indexer_l1_ring(
                            Path(artifact.output_dir) / "ptoas", key_pairs=args.indexer_key_pairs)
                        save(args.output / "indexer_l1_ring_codegen_audit.json", audit_result)
                        (args.output / "indexer_l1_ring_stage.cpp.txt").write_text(stage_cpp + "\n")
                        (args.output / "indexer_l1_ring_loop.cpp.txt").write_text(loop_cpp + "\n")
                        stage("indexer_l1_ring_codegen_audit_passed_before_execution")
                    if args.bypass_wait or args.cached_wait:
                        # 所有编译扩展先入栈，再解析artifact；审计不得提前缓存默认等待版本。
                        from signal_polling import audit_wait_artifact

                        artifact = module.entry._resolve_kernel_artifact(tuple(entry_args), {"config": options})
                        report["signal_polling_artifact_audit"] = audit_wait_artifact(
                            Path(artifact.output_dir), args.output / "signal_polling",
                            args.wait_backoff_ticks, cached=args.cached_wait,
                            ramp_backoff=args.ramp_wait)
                        save(args.output / "signal_polling_artifact_audit.json",
                             report["signal_polling_artifact_audit"])
                        stage("signal_polling_artifact_audit_passed_before_execution")
                    entries["single_spmd"]()
                stream.synchronize()
                generations += 1
                report["eager_exact"] = check(reference)
                stage("both_eager_entries_passed")
                # 预编译完成后只捕获 PTO 调用。reset 不进图，不执行 Native 对照。
                for side in entries:
                    reset()
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        entries[side]()
                    graphs[side] = graph
                    reset()
                    graph.replay()
                    stream.synchronize()
                    if side == "single_spmd":
                        generations += 1
                    report[side + "_graph_exact"] = check(reference)
                stage("both_graph_replays_passed")
                if args.paired_qproj_spmd_only or args.paired_kv_signal_spmd_only:
                    from paired_row_dependency_perf import compare

                    paired_folder = "paired_kv_signal_spmd" if args.paired_kv_signal_spmd_only else "paired_qproj_spmd"
                    paired_label = "kv_touch_signal_spmd" if args.paired_kv_signal_spmd_only else "qproj_l1_columns_spmd"
                    report.update(warmup=2, iterations=0, l1_windows_per_side=3)
                    paired, extra_generations = compare(
                        args=args, call=call, options=options, graphs=graphs, stream=stream,
                        reset=reset, check=check, reference=reference, export_window=export_window,
                        statistics_us=statistics_us, reference_row_dependencies=True,
                        candidate_label=paired_label, folder=paired_folder)
                    generations += extra_generations
                    report[paired_folder] = paired
                    progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
                    report["signal_generations"] = generations
                    report["progress_passed"] = torch.equal(progress, progress_expectation(layout, generations))
                    report["guards"] = guard_checks(fixture)
                    guard = prepared.workspace_guard_bytes
                    report["workspace_guards_passed"] = bool(
                        (prepared.workspace_allocation[:guard].cpu() == 37).all()
                        and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
                    if not (report["progress_passed"] and report["workspace_guards_passed"]
                            and checks_passed(report["guards"])):
                        raise AssertionError("候选同进程对照的信号或保护区不符")
                    report.update(passed=True, performance_measured=True, graph_replay_validated=True,
                                  l1_incore=paired["l1_incore"],
                                  scope="现选row-deps单SPMD与候选的同进程三窗L1：" + paired_label)
                    stage("complete")
                    return
                if args.paired_selected_spmd:
                    from paired_row_dependency_perf import compare

                    report["paired_selected_spmd"], extra_generations = compare(
                        args=args, call=call, options=options, graphs=graphs, stream=stream,
                        reset=reset, check=check, reference=reference, export_window=export_window,
                        statistics_us=statistics_us)
                    generations += extra_generations
                    stage("paired_selected_spmd_complete")
                if args.metadata_replay_only:
                    from dynamic_metadata_replay import validate

                    before_builds = sorted(path.name for path in Path(os.environ["PYPTO_PROG_BUILD_DIR"]).iterdir())
                    report["scope"] = "正式第4层权重；同batch/S6/容量/地址的完整CSA动态metadata图重放"
                    report["warmup"] = report["iterations"] = report["l1_windows_per_side"] = 0
                    report["metadata_replay"], extra_generations = validate(
                        config=config, layer=layer, adapter=adapter, weights=weights, fixture=fixture,
                        call=call, prepared=prepared, entries=entries, graphs=graphs, stream=stream,
                        batch=batch, history=history, device=device, directory=args.output / "metadata_replay",
                        extra_histories=args.metadata_history)
                    generations += extra_generations
                    after_builds = sorted(path.name for path in Path(os.environ["PYPTO_PROG_BUILD_DIR"]).iterdir())
                    report["metadata_replay"]["no_new_compilation_units"] = before_builds == after_builds
                    progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
                    report["signal_generations"] = generations
                    report["progress_passed"] = torch.equal(progress, progress_expectation(layout, generations))
                    report["guards"] = guard_checks(fixture)
                    guard = prepared.workspace_guard_bytes
                    report["workspace_guards_passed"] = bool(
                        (prepared.workspace_allocation[:guard].cpu() == 37).all()
                        and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
                    if not (report["progress_passed"] and report["workspace_guards_passed"]
                            and checks_passed(report["guards"]) and before_builds == after_builds):
                        raise AssertionError("动态metadata检查结束后的信号/保护区/编译次数不符")
                    report.update(passed=True, performance_measured=bool(args.paired_selected_spmd),
                                  graph_replay_validated=True)
                    stage("complete")
                    return
                start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                samples = {side: [] for side in entries}
                stamps = []
                for iteration in range(args.warmup + args.iters):
                    # AB/BA 交替，避免先测/后测顺序与温度趋势始终偏向一侧。
                    order = list(entries) if iteration % 2 == 0 else list(reversed(entries))
                    for side in order:
                        reset()
                        start.record()
                        graphs[side].replay()
                        end.record()
                        stream.synchronize()
                        if side == "single_spmd":
                            generations += 1
                        stamp = start.recorded_time()
                        if stamps and stamp <= stamps[-1]:
                            raise ValueError("设备事件时间戳没有更新")
                        stamps.append(stamp)
                        elapsed = start.elapsed_time(end) * 1000
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise ValueError("设备事件耗时无效")
                        if iteration >= args.warmup:
                            samples[side].append(elapsed)
                report["event_replay"] = {side: statistics_us(values) for side, values in samples.items()}
                report["event_scope"] = "图外NPU Event包住单次图重放；不含fixture恢复，DFX窗口关闭"
                stage("event_timing_complete")
                window = 0
                for sample in range(args.windows):
                    order = list(entries) if sample % 2 == 0 else list(reversed(entries))
                    for side in order:
                        reset()
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        if side == "single_spmd":
                            generations += 1
                        source = args.output / "dfx"
                        if window:
                            source /= f"window_{window}"
                        destination = args.output / "l1" / side / f"step_{sample + 1}"
                        record = export_window(source, destination, side)
                        record.update(side=side, step=sample + 1, dfx_window=window)
                        report["l1_windows"].append(record)
                        window += 1
                        if sample == args.windows - 1:
                            report[side + "_final_exact"] = check(reference)
                        stage(f"l1_{side}_{sample + 1}_complete")
                if args.pipe_profile:
                    # kernel 模式没有 Simpler enable_pmu 入口；用已有 torch_npu
                    # 外部采样，保留完整持久核记录。此处结果不进入前面的 L1/Event 时间。
                    report["pipe_profiles"] = {}
                    for side in entries:
                        destination = args.output / "pipe_profile" / side
                        with torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                        torch_npu.profiler.ProfilerActivity.NPU],
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
                            experimental_config=torch_npu.profiler._ExperimentalConfig(
                                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
                        ) as trace:
                            for sample in range(3):
                                reset()
                                stream.synchronize()
                                with torch.profiler.record_function("CSA_" + side):
                                    graphs[side].replay()
                                    stream.synchronize()
                                if side == "single_spmd":
                                    generations += 1
                                trace.step()
                        traces = list(destination.glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
                        if len(traces) != 1:
                            raise ValueError(f"外部 profiling 未生成唯一 trace：{side}")
                        report["pipe_profiles"][side] = {"trace": str(traces[0]),
                            "scope": "完整持久核外部 PMU，包含外围调度；不替代 L1 incore 计时",
                            "steps": 3, "post_profile_exact": check(reference)}
                        stage("pipe_profile_" + side + "_complete")
                progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
                expected = progress_expectation(layout, generations)
                report["signal_generations"] = generations
                report["progress_passed"] = torch.equal(progress, expected)
                if args.task_completion:
                    all_signals = prepared.sections["progress"].cpu()
                    counters = [{"task": row["task"], "actual": int(all_signals[row["offset_words"]]),
                                 "expected": generations * row["members"]}
                                for row in layout["task_completion"]["tasks"]]
                    report["task_counters"] = counters
                    report["task_counters_passed"] = all(row["actual"] == row["expected"] for row in counters)
                    if not report["task_counters_passed"]:
                        raise AssertionError("任务计数器代际检查失败")
                if args.stage_clock:
                    report["stage_clock"] = export_stage_clock(
                        prepared.sections["progress"], layout, args.output / "stage_clock")
                if args.internal_wait_clock:
                    from internal_wait_clock import export_clock

                    report["internal_wait_clock"] = export_clock(
                        prepared.sections["progress"], args.output / "internal_wait_clock")
                if args.qk_phase_clock:
                    from qk_phase_clock import export_clock as export_phases

                    report["qk_phase_clock"] = export_phases(
                        prepared.sections["progress"], args.output / "qk_phase_clock")
                if args.indexer_phase_clock:
                    from indexer_phase_clock import export_clock as export_indexer_phases

                    report["indexer_phase_clock"] = export_indexer_phases(
                        prepared.sections["progress"], args.output / "indexer_phase_clock")
                report["guards"] = guard_checks(fixture)
                guard = prepared.workspace_guard_bytes
                report["workspace_guards_passed"] = bool(
                    (prepared.workspace_allocation[:guard].cpu() == 37).all()
                    and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
                if not (report["progress_passed"] and report["workspace_guards_passed"]
                        and checks_passed(report["guards"])):
                    raise AssertionError("性能结束后的信号或保护区检查失败")
            report["l1_incore"] = {
                side: statistics_us([r["incore_span_us"] for r in report["l1_windows"] if r["side"] == side])
                for side in entries}
            report["l1_scope"] = "min(incore start)至max(incore end)，保留核内等待和中间空隙，排除外围调度首尾"
            report["comparison"] = {}
            for metric in ("l1_incore", "event_replay"):
                baseline = report[metric]["dynamic_pto"]["median_us"]
                candidate = report[metric]["single_spmd"]["median_us"]
                report["comparison"][metric] = {"dynamic_pto_us": baseline, "single_spmd_us": candidate,
                                                "speedup": baseline / candidate,
                                                "latency_change_pct": (candidate / baseline - 1) * 100}
            report.update(passed=True, performance_measured=True, graph_replay_validated=True)
            from performance_acceptance import evaluate

            report["performance_target"] = evaluate(report)
            stage("complete")
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        stage("failed")
        raise


if __name__ == "__main__":
    main()
