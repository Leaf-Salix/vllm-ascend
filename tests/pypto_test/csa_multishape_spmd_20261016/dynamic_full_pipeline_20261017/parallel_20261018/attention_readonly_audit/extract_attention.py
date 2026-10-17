#!/usr/bin/env python3
"""只读提取已有长 B16 作业的 Attention 代码与体积证据，不编译、不导入设备库。

输入限定为同一次 current_clock_h131072_b16_v1 中的原分 task 和融合产物；
输出仅为本目录 JSON 与文本摘录。PTO 指令数是静态文本位置数量，不是动态执行
次数；C++ 取各自 AIV 函数的 Attention 初始化至最后循环携带 m 的回写，避免
把全 CSA 的其他数学或同文件的 AIC 函数算进 Attention。二进制只读文件大小，
不计算/输出 hash，不将整层二进制大小声称为 Attention 独立机器码大小。
"""

import argparse
import collections
import json
import re
from pathlib import Path


DEFAULT_RUN = Path(
    "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/"
    "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/"
    "current_clock_h131072_b16_v1"
)


def cpp_function(source, name):
    """局部编号 vN 在不同函数中复用，必须先限制函数范围再解析常量。"""
    start = source.index("static __aicore__ void " + name + "(")
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor], start


def cpp_attention(source, name):
    function, offset = cpp_function(source, name)
    start = function.index("// pto: %m_idx_inline")
    matches = list(re.finditer(r"// pto: %m_after_inline\d+__rv_v0_mv", function))
    end = function.index(";", function.index("TMOV(", matches[-1].start())) + 1
    text = function[start:end]
    counts = collections.Counter(
        re.findall(r"\b(T[A-Z][A-Z0-9_]*|set_flag|wait_flag|pipe_barrier|wait_flag_dev)\s*(?:<[^;]*?>)?\(", text)
    )
    barriers = []
    for match in re.finditer(r"pipe_barrier\(PIPE_ALL\);", text):
        barriers.append(source[: offset + start + match.start()].count("\n") + 1)
    return text, {
        "original_lines": [source[: offset + start].count("\n") + 1, source[: offset + end].count("\n") + 1],
        "bytes": len(text.encode()),
        "lines": len(text.splitlines()),
        "intrinsic_sites": dict(sorted(counts.items())),
        "pipe_all_original_lines": barriers,
        "barrier_sites": dict(collections.Counter(re.findall(r"pipe_barrier\(([^;]+)\);", text))),
        "local_wait_sites": dict(collections.Counter(re.findall(r"wait_flag\(([^;]+)\);", text))),
    }


def pto_attention(source, name):
    lines = source.splitlines()
    func_start = next(i for i, line in enumerate(lines) if "func.func @" + name + "(" in line)
    func_end = next((i for i in range(func_start + 1, len(lines)) if re.match(r"  func.func ", lines[i])), len(lines))
    start = next(i for i in range(func_start, func_end) if re.search(r"%qk_core_inline\d+__ssa_v0 =", lines[i]))
    end = max(i for i in range(start, func_end) if re.search(r"pto.tmov.*outs\(%m_after_inline", lines[i])) + 2
    text = "\n".join(lines[start:end]) + "\n"
    counts = collections.Counter(re.findall(r"(?m)^\s*(?:%[^=\n]+ = )?((?:pto|arith|scf)\.[a-z_.]+)", text))
    semantic = re.sub(r' loc\("[^"\n]*"[^\n]*\)$', "", text, flags=re.M)
    sync = []
    for i in range(start, end):
        if re.search(r"pto.sync.(set|wait) ", lines[i]):
            sync.append({"line": i + 1, "instruction": lines[i].strip().split(" loc(")[0]})
    return text, {
        "original_lines": [start + 1, end],
        "bytes": len(text.encode()),
        "lines": len(text.splitlines()),
        "bytes_without_locations": len(semantic.encode()),
        "operation_sites": dict(sorted(counts.items())),
        "sync_sites": sync,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build = args.run / "build_output"
    entries = {
        "task": (build / "_decode_csa_tp1_layer_0leidxnz", "qk_pv_aiv", "qk_pv"),
        "fused": (build / "decode_csa_spmd_jqlk5eey", "vector_schedule", "dynamic_csa_full"),
    }
    result = {"run": str(args.run), "read_only": True, "compiled": False, "device_submitted": False, "sides": {}}
    for label, (root, function, pto_name) in entries.items():
        cpp = root / "kernels/aiv" / (function + ".cpp")
        pto = root / "ptoas" / (pto_name + ".pto")
        cpp_text, cpp_stats = cpp_attention(cpp.read_text(), function)
        pto_text, pto_stats = pto_attention(pto.read_text(), function)
        (args.output / (label + "_attention.cpp.txt")).write_text(cpp_text)
        (args.output / (label + "_attention.pto.txt")).write_text(pto_text)
        # 现成清单含其他字段；只读取名称、相对路径和实际字节大小，不输出 hash。
        manifest = json.loads((root / "binary_manifest.json").read_text())
        selected = [x for x in manifest["kernels"] if x["name"] in (function, "cube_schedule", "qk_pv_aic")]
        binaries = [{"name": x["name"], "path": str(root / x["binary"]["path"]),
                     "bytes": (root / x["binary"]["path"]).stat().st_size} for x in selected]
        result["sides"][label] = {
            "cpp": {"path": str(cpp), "whole_file_bytes": cpp.stat().st_size, **cpp_stats},
            "pto": {"path": str(pto), "whole_file_bytes": pto.stat().st_size, **pto_stats},
            "binaries": binaries,
        }
    fc = collections.Counter(result["sides"]["fused"]["pto"]["operation_sites"])
    tc = collections.Counter(result["sides"]["task"]["pto"]["operation_sites"])
    result["pto_site_difference"] = {"fused_extra": dict(fc - tc), "task_extra": dict(tc - fc)}
    tile_sequences = {}
    for side in entries:
        text = (args.output / (side + "_attention.pto.txt")).read_text()
        sequence = re.findall(r"(?m)^\s*(?:%[^=\n]+ = )?(pto\.t[a-z_]+)", text)
        # 原 task 在 SWA/compact 互斥分支各发一个同形零填充，融合在分支前
        # 发一次；删除的是对比文本项，不删除任何生产执行代码。
        tile_sequences[side] = [x for x in sequence if x != "pto.texpands"]
    result["tile_operation_order"] = {
        "exclude": "仅排除 pto.texpands 的互斥分支/初始化文本位置",
        "counts": {side: len(seq) for side, seq in tile_sequences.items()},
        "equal": tile_sequences["task"] == tile_sequences["fused"],
        "sequence": tile_sequences["fused"],
        "boundary": "操作名序列相同不单独证明操作数、地址、数学逐bit或机器码等价",
    }
    summary = json.loads((args.run / "phase_summary.json").read_text())
    result["existing_clock"] = {
        "eligible_for_performance": summary["eligible_for_performance"],
        "rounds": [{"round": r["round"], "label": r["label"], "attention": {
            n: {k: r["phases"][n][k] for k in ("name", "interval_us", "wait_us", "ffts_issue_us", "ffts_issue_calls")}
            for n in ("38", "39")}} for r in summary["rounds"]],
    }
    invocation = json.loads((args.run / "result.json").read_text())
    result["invocation"] = {k: invocation[k] for k in ("passed", "history", "batch", "query_tokens", "policies")}
    tokens = invocation["batch"] * invocation["query_tokens"]
    result["long_b16_work_model"] = {
        "assumption": "同次H131072/B16/S6，全部query的5块有效；每核40个FFTS wait与该模型相符",
        "tokens": tokens, "queries_per_core_group": tokens // 24,
        "aic_ticks": tokens // 24 * 5 + 2,
        "aiv_ticks": tokens // 24 * 5 + 3,
        "valid_blocks_per_core_group": tokens // 24 * 5,
        "ffts_waits_per_aiv": tokens // 24 * 5 * 2,
        "compressed_scalar_gather_iterations_per_aiv": tokens // 24 * 4 * 64,
        "kv_source_logical_bytes_whole_card": tokens * 640 * 512 * 2,
        "kv_ring_store_logical_bytes_whole_card": tokens * 640 * 512 * 2,
        "score_read_logical_bytes_whole_card": tokens * 640 * 64 * 4,
        "probability_store_logical_bytes_whole_card": tokens * 640 * 64 * 2,
        "pv_read_logical_bytes_whole_card": tokens * 5 * 64 * 512 * 4,
        "query_geometry_scans_per_aiv": tokens // 24 * 5,
        "query_geometry_request_visits_per_aiv": tokens // 24 * 5 * invocation["batch"],
        "query_geometry_scalar_loads_per_aiv": tokens // 24 * 5 * (3 * invocation["batch"] + 2),
        "boundary": "逻辑数据量不等于HBM实际流量；Scalar读取可命中缓存，不能据计数折算收益",
    }
    result["limitations"] = [
        "静态指令位置计数包括互斥分支，不是动态执行次数或周期。",
        "融合二进制包括完整CSA及phase/FFTS诊断，原task二进制仅Attention；不能据体积判定icache失速。",
        "没有Attention机器码符号范围；不能精确分割融合二进制中的Attention字节。",
        "C++摘录排除前置GM phase获取和末尾发布；PTO摘录包括阶段内联的请求几何。",
        "原始manifest的hash字段未参与计算或输出。",
    ]
    output = args.output / "evidence.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "pto_site_difference": result["pto_site_difference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
