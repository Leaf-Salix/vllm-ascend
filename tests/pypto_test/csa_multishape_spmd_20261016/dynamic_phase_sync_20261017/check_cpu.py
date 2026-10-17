# SPDX-License-Identifier: Apache-2.0
"""动态阶段信号的 CPU 编译检查；不执行设备、不代替完整 CSA 验证。

动态 T 的 AIC 按 16 行轮转生产，双 AIV 按 8 行消费；消费者根据自己的
实际行范围推算前驱。只检查动态源与单 MIX lowering，没有性能结论。
"""

import argparse
import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import RunConfig

REPOSITORY = Path(__file__).resolve().parents[4]
PACKAGE = REPOSITORY / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
spec = importlib.util.spec_from_file_location(
    "_csa_dynamic_sync_probe", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from _csa_dynamic_sync_probe.phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners

T = pl.dynamic("PROBE_TOKENS")
WORDS = pl.dynamic("PROBE_SIGNAL_WORDS")


@pl.jit.incore
def cube(
    x: pl.Tensor[[T, 16], pl.BF16],
    w: pl.Tensor[[16, 16], pl.BF16],
    mm: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    result: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    rows = pl.tensor.dim(x, 0)
    blocks = (rows + 15) // 16
    for ordinal in pl.range(worker, blocks, 24):
        row = ordinal * 16
        valid = pl.min(16, rows - row)
        left = pl.load(x, [row, 0], [16, 16], valid_shape=[valid, 16], target_memory=pl.Mem.Mat)
        right = pl.load(w, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        value = pl.matmul(left, right)
        mm = pl.store(value, [row, 0], mm)
    signal = publish_phase(signal, 1, worker, 72, generation)
    # 每个 AIV 即使没有有效行也会发布阶段 2；此等待不构成阶段内循环依赖。
    peer = 24 + ((worker + 1) % 24) * 2
    signal = wait_producer(signal, 2, peer, 72, generation)
    return mm, result, signal


@pl.jit.incore
def vector(
    mm: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    result: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = 24 + worker
    signal, generation = begin_invocation(signal, core)
    rows = pl.tensor.dim(result, 0)
    blocks = (rows + 7) // 8
    for ordinal in pl.range(worker, blocks, 48):
        row = ordinal * 8
        valid = pl.min(8, rows - row)
        signal = wait_row_owners(signal, 1, 72, generation, row, valid, 16, 24, 0, 1)
        value = pl.load(mm, [row, 0], [8, 16], valid_shape=[valid, 16])
        result = pl.store(pl.add(value, 1.0), [row, 0], result)
    signal = publish_phase(signal, 2, core, 72, generation)
    return mm, result, signal


@pl.jit.inline
def group(
    x: pl.Tensor[[T, 16], pl.BF16],
    w: pl.Tensor[[16, 16], pl.BF16],
    mm: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    result: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.cluster(name_hint="dynamic_phase_sync_probe"):
        mm, result, signal = cube(x, w, mm, result, signal)
        mm, result, signal = vector(mm, result, signal)
    return mm, result, signal


@pl.jit
def entry(
    x: pl.Tensor[[T, 16], pl.BF16],
    w: pl.Tensor[[16, 16], pl.BF16],
    mm: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    result: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.spmd(24, sync_start=True):
        mm, result, signal = group(x, w, mm, result, signal)
    return mm, result, signal


def check_owner_formula():
    """与逐行覆盖集合独立比较，不以复写相同 ceil_div 公式作为断言。"""
    comparisons = 0
    for workers in (1, 3, 16, 24, 48):
        for tile in (2, 8, 16, 32):
            for first in (0, 1, 7, 17, 63, 129):
                for count in (0, 1, 2, 17, 96, 257):
                    expected = {(row // tile) % workers for row in range(first, first + count)}
                    first_tile = first // tile
                    end_tile = (first + count + tile - 1) // tile
                    actual = set() if not count else {
                        (first_tile + i) % workers for i in range(min(end_tile - first_tile, workers))
                    }
                    assert actual == expected, (workers, tile, first, count, actual, expected)
                    comparisons += 1
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device_execution": False, "full_csa_validated": False}
    try:
        # 保存本次实际源码，后续编辑不得反写历史编译证据；不生成 hash。
        (args.output / "probe_source.py").write_text(Path(__file__).read_text())
        for name in ("phase_sync.py", "signals.py"):
            (args.output / name).write_text((PACKAGE / name).read_text())
        report["owner_set_comparisons"] = check_owner_formula()
        program = entry.specialize()
        logical = python_print(program)
        (args.output / "symbolic_entry.py").write_text(logical)
        assert "PROBE_TOKENS" in logical and "PROBE_SIGNAL_WORDS" in logical
        compiled = entry.compile(config=RunConfig(
            platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
            dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO,
        ))
        output = Path(compiled.output_dir)
        orchestration = "\n".join(p.read_text() for p in (output / "orchestration").glob("*.cpp"))
        calls = re.findall(r"\brt_submit_\w+\s*\(", orchestration)
        report.update(
            output_dir=str(output), submission_calls=calls,
            single_mix_submission=len(calls) == 1 and "MixedKernels" in orchestration,
            sync_start="set_require_sync_start(true)" in orchestration,
            full_codegen_passed=True,
        )
        assert report["single_mix_submission"] and report["sync_start"]
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
