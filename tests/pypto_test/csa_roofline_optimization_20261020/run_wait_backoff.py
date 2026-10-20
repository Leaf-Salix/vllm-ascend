# SPDX-License-Identifier: Apache-2.0
"""仅在本测试进程比较 GM 信号轮询的 50/100/150 tick 固定退避。

整体流程：
1. 复用已验证的完整 CSA、同址输入/状态、图重放与五窗交替 L1 测试。
2. 候选和参考加载同一份正式动态 SPMD 的冻结源码；参考始终为 100 tick。
3. 仅在现有编译回调内改变等待 helper 的计时常量，不改数学、分核、
   DCCI/volatile 读取、发布端或 payload fence，也不叠加 Top-K 压缩候选。
4. 核对真实 CCEC 输入里的常量，防止复用未应用变换的旧产物。

本文件不修改正式 wait_codegen.py 或共享安装。每次运行使用独立构建目录，
关闭磁盘编译缓存。50 与 150 分别配同轮 100 对照，不拿跨轮耗时拼接收益。
"""

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


DELAY_PREFIX = "while ((uint64_t)(get_sys_cnt() - csa_wait_start) < "
DEFAULT_DELAY = DELAY_PREFIX + "100)"


def rewrite_ticks(source, ticks, original_rewrite):
    """从原 100 tick 变换严格派生，只允许一个计时常量发生变化。"""
    if ticks not in (50, 100, 150):
        raise ValueError("本轮只验证 50、100、150 tick")
    rewritten, audit = original_rewrite(source)
    if rewritten.count(DEFAULT_DELAY) != 1 or audit["backoff_syscnt_ticks"] != 100:
        raise AssertionError("原等待实现已改变，不能静默套用旧实验")
    requested = DELAY_PREFIX + str(ticks) + ")"
    changed = rewritten.replace(DEFAULT_DELAY, requested)
    if changed.replace(requested, DEFAULT_DELAY) != rewritten:
        raise AssertionError("变换范围超出单个退避常量")
    audit = dict(audit)
    audit.update(
        backoff_syscnt_ticks=ticks,
        initial_backoff_syscnt_ticks=ticks,
        parameter_only=True,
        device_validated=False,
    )
    return changed, audit


def audit_actual_sources(paths, ticks):
    """检查 kernels 目录的实际 CCEC 输入，不能用 ptoas 目录的中间产物代替。"""
    if len(paths) != 2:
        raise AssertionError(f"本次编译应新建一对 AIC/AIV 实际源码，得到 {len(paths)} 份")
    rows = []
    for path in sorted(paths):
        text = path.read_text()
        if text.count(DELAY_PREFIX + str(ticks) + ")") != 1:
            raise AssertionError(f"实际 CCEC 源码的退避参数不符：{path}")
        if text.count("CSA_WAIT_LOCAL_GM(") < 2 or "pto::comm::TWAIT(" in text:
            raise AssertionError(f"实际 CCEC 源码仍混有未替换等待：{path}")
        required = (
            "dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);",
            "reinterpret_cast<volatile __gm__ int32_t*>(signal.data())",
            "if (*ptr >= expected) break;",
        )
        if not all(item in text for item in required):
            raise AssertionError(f"信号读侧合同缺失：{path}")
        rows.append(dict(path=str(path), ticks=ticks, wait_calls=text.count("CSA_WAIT_LOCAL_GM(") - 1))
    return rows


def normalize_comments(text):
    """只忽略 IR 行号等注释及空白，数学指令和常量均保留比较。"""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-wait-ticks", type=int, choices=(50, 150), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spmd-source", type=Path, required=True)
    parser.add_argument("--source-matrix-reference", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if args.spmd_source.resolve() != args.source_matrix_reference.resolve():
        raise ValueError("退避实验要求候选和参考使用同一个冻结算子目录")
    if args.output.exists():
        raise FileExistsError(args.output)
    if "--wait-backoff" not in forwarded:
        raise ValueError("必须显式开启已有 wait-backoff 编译回调")

    driver = here.parent / (
        "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/"
        "parallel_20261018/mgather_physical_plan_retest_20261019"
    )
    sys.path.insert(0, str(driver))
    diagnostics = importlib.import_module("compile_diagnostics")
    original_probe = diagnostics.compilation_probe
    audit_rows = []

    @contextlib.contextmanager
    def parameter_probe(directory, *, wait_backoff=False, phase_clock=False):
        directory = Path(directory)
        if not wait_backoff or phase_clock:
            raise ValueError("退避对照只允许无插桩的完整等待变换")
        if directory.name == "codegen_probe":
            role, ticks = "candidate", args.candidate_wait_ticks
        elif directory.name == "spmd_reference_codegen":
            role, ticks = "reference", 100
        else:
            raise ValueError(f"未识别的编译角色：{directory}")
        # 此时 run_frozen_full 已绑定公共冻结源码；提前导入会破坏其隔离检查。
        wait_module = importlib.import_module(
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen"
        )
        original_rewrite = wait_module.rewrite_wait_source
        build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
        before = set(build.glob("*decode_csa_spmd_*/kernels/*/*.cpp"))

        def parameter_rewrite(source):
            return rewrite_ticks(source, ticks, original_rewrite)

        wait_module.rewrite_wait_source = parameter_rewrite
        try:
            with original_probe(directory, wait_backoff=True, phase_clock=False):
                yield
        finally:
            wait_module.rewrite_wait_source = original_rewrite
        generated = set(build.glob("*decode_csa_spmd_*/kernels/*/*.cpp")) - before
        actual = audit_actual_sources(generated, ticks)
        row = dict(role=role, ticks=ticks, actual_cc_ec_inputs=actual, probe=str(directory))
        (directory / "actual_wait_audit.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n"
        )
        audit_rows.append(row)

    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    diagnostics.compilation_probe = parameter_probe
    sys.argv = [
        str(here / "run_frozen_full.py"),
        "--output", str(args.output),
        "--spmd-source", str(args.spmd_source),
        "--source-matrix-reference", str(args.source_matrix_reference),
        *forwarded,
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        report_path = args.output / "result.json"
        report = json.loads(report_path.read_text())
        if not report["passed"]:
            raise AssertionError("完整精度/图重放/状态检查未通过")
        if sorted(row["role"] for row in audit_rows) != ["candidate", "reference"]:
            raise AssertionError("候选和 100 tick 参考未分别完成独立编译")
        if not report["compiler_comparison"]["same_operator_source"]:
            raise AssertionError("退避对照混入了其他算子源码差异")
        originals = [Path(row["probe"]) / "original.cpp" for row in audit_rows]
        cpp_equal = normalize_comments(originals[0].read_text()) == normalize_comments(originals[1].read_text())
        comparison = dict(
            passed=cpp_equal,
            source_only_difference="GM 信号未就绪后的固定退避常量",
            original_cpp_equal_ignoring_comments_and_whitespace=cpp_equal,
            roles=audit_rows,
            statistics_us=report["source_matrix"]["statistics_us"],
        )
        (args.output / "WAIT_BACKOFF_COMPARISON.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
        )
        if not cpp_equal:
            raise AssertionError("变换前 C++ 仍有数学/调度差异，不能当作纯退避结果")
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    finally:
        diagnostics.compilation_probe = original_probe
        if args.output.is_dir():
            (args.output / "WAIT_BACKOFF_REQUEST.json").write_text(
                json.dumps(dict(
                    candidate_ticks=args.candidate_wait_ticks,
                    reference_ticks=100,
                    source=str(args.spmd_source.resolve()),
                    cache_disabled=True,
                    completed_codegen_audits=audit_rows,
                ), ensure_ascii=False, indent=2) + "\n"
            )
            (args.output / Path(__file__).name).write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
