# SPDX-License-Identifier: Apache-2.0
"""用完整 CSA 的相同真实中间值对照原 load3d 与候选 load2d。

只做一组有硬件依据的搬运消融。两侧调用原来的完整 Score 数学，均比较
完整 pair/score allocation，分别采三个 PMU step。阶段图外 Event 只用于
筛选，不能替代完整 CSA L1；两侧未交错采样，不能宣称小差值稳定成立。
"""

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

from load2d_hook import patch_score


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(prepared, compile_options, output):
    output.mkdir(parents=True, exist_ok=False)
    baseline_dir = HERE.parent / "indexer_score_hardware_evidence"
    original = load("score_load3d_reference", baseline_dir / "score_kernel.py")
    candidate = load("score_load2d_candidate", HERE / "candidate_score_kernel.py")
    profiler = load("score_existing_profiler", baseline_dir / "profile_score.py")
    constants = prepared.policy.constexprs()
    if not (constants["QUERY_GROUP"] == 6 and constants["QK_PANEL"] == 64
            and constants["SCORE_TILE"] == 1024 and constants["BALANCED"]):
        raise ValueError("本轮诊断只验证 G6/N64/1024 balanced，不扩展其余策略")
    report = dict(passed=False, diagnostic_only=True, eligible_for_full_csa_performance=False,
                  variants={}, input_origin="同一完整CSA A→B→A后的实际中间值")
    previous = sys.modules.get("score_kernel")
    try:
        for label, entry in (("load3d", original.score_entry), ("load2d", candidate.load2d_score_entry)):
            proxy = types.ModuleType("score_kernel")
            proxy.score_entry = entry
            sys.modules["score_kernel"] = proxy
            context = patch_score(output / "codegen_load2d") if label == "load2d" else contextlib.nullcontext()
            with context:
                profiler.measure(prepared, compile_options, output / label)
            detail = json.loads((output / label / "result.json").read_text())
            report["variants"][label] = dict(passed=detail["passed"], event_us=detail["event_us"],
                                              result=str(output / label / "result.json"))
        audit = json.loads((output / "codegen_load2d/audit.json").read_text())
        if audit["count"] != 16:
            raise AssertionError("实际编译未完整应用16处WS搬运替换")
        report.update(passed=True, codegen=audit)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if previous is None:
            sys.modules.pop("score_kernel", None)
        else:
            sys.modules["score_kernel"] = previous
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return dict(passed=True, diagnostic_only=True, result=str(output / "result.json"))
