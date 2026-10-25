# SPDX-License-Identifier: Apache-2.0
"""同MIX内以24个AIV清零signal，与前半AIV的无信号依赖RMS并行。"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

from prepare import hashes, replace_once


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/internal_single_f72a"
    target = here / "candidates/init_overlap_rms_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    abi = (target / "runtime_abi.py").read_text()
    abi = replace_once(abi, "VECTORS = 48\n", "VECTORS = 48\nSIGNAL_INIT_WORKERS = VECTORS // 2\n")
    (target / "runtime_abi.py").write_text(abi)
    for name in ("entry.py", "vector_schedule.py"):
        text = (target / name).read_text()
        text = replace_once(text, "\n    VECTORS,\n", "\n    SIGNAL_INIT_WORKERS,\n    VECTORS,\n")
        old = "[VECTORS, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS]"
        new = "[SIGNAL_INIT_WORKERS, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // SIGNAL_INIT_WORKERS]"
        expected = 3 if name == "entry.py" else 1
        assert text.count(old) == expected
        text = text.replace(old, new)
        if name == "vector_schedule.py":
            start = text.index("    # Exactly the original 48 contiguous, disjoint 11520-word ranges.\n")
            end = text.index("    if BALANCED and QUERY_GROUP == 2:\n", start)
            replacement = """    # 无条件初始化仍在单MIX内：后24个AIV清理完整signal，前半可先做RMS。
    # hc_widen_rms_range不读写signal，只读根输入并写本核所有的x_fp32/inv。
    # 所有signal访问/阶段发布仍在全核硬屏障之后，任意T都执行同一协议。
    pl.system.cacheinvalid()
    pl.system.fence()
    if worker >= VECTORS - SIGNAL_INIT_WORKERS:
        clear_worker = worker - (VECTORS - SIGNAL_INIT_WORKERS)
        zero = pl.tile.full(
            [1, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // SIGNAL_INIT_WORKERS], dtype=pl.INT32, value=0
        )
        pl.store(zero, [clear_worker, 0], signal_init_matrix)
    tokens = pl.tensor.dim(x, 0)
    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)
    # 保留硬屏障前的缓存维护和流水完成，发布前须同时完成清零和输入写回。
    pl.system.cacheinvalid()
    pl.system.fence()
    pl.system.syncall(core_type=pl.KernelType.MIX)
    pl.system.cacheinvalid()
    signal, generation = begin_invocation(signal, core)
"""
            text = text[:start] + replacement + text[end:]
        ast.parse(text)
        (target / name).write_text(text)
    changed = [str(target / name) for name in ("entry.py", "vector_schedule.py", "runtime_abi.py")]
    subprocess.run(["ruff", "format", *changed], check=True)
    subprocess.run(["ruff", "check", "--fix", *changed], check=True)
    total_words = 60 * 72 * 128
    chunk = total_words // 24
    assert total_words % 24 == 0
    intervals = [(i * chunk, (i + 1) * chunk) for i in range(24)]
    assert intervals[0][0] == 0 and intervals[-1][1] == total_words
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    report = {
        "status": "CPU_gate_pending",
        "candidate": str(target),
        "source_hashes": hashes(target),
        "production_eligible_if_validated": True,
        "adopted": False,
        "change": "signal全量清零改由AIV24..47执行；原RMS提前到硬MIX屏障前，与清零独立并行",
        "preserved": [
            "所有输入档位同一单MIX初始化协议，无独立任务或外部signal",
            "完整2211840B清零、512B槽距、硬MIX屏障及后续phase发布/等待",
            "RMS原32个K512归约次序、原worker所有权和尾行处理；所有其他数学未改",
            "若T>192后半AIV也有RMS工作，则先清零再执行其原RMS，仍全部到屏障后才读signal",
        ],
        "CPU_ownership": {"writers": 24, "words_per_writer": chunk, "coverage": "552960个word恰好写一次"},
        "risks_to_validate": ["90KiB清零tile与现有UB生命周期", "编译器保留硬屏障与信号时序", "跨重放精度及状态"],
    }
    (here / "init_overlap_rms_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
