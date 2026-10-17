# SPDX-License-Identifier: Apache-2.0
"""无设备/无样例shape编译未启用策略，并对照默认源码的完整内核C++。"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = Path(
    "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
)
CANDIDATE = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
REFERENCE = D / "parallel_20261018/baseline_source"


def normalize_cpp(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r"\s+", " ", source).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("g6_768", "g2_1024"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(D))
    report = dict(passed=False, variant=args.variant, device_execution=False, sample_shapes_supplied=False)
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        from pypto import passes  # noqa: I001 - 依赖隔离激活后再延迟导入验证入口
        from pypto.runtime import CompileOptions
        from source_loader import load_package
        from check_cpu import audit_codegen

        controls = dict(
            QUERY_GROUP=6 if args.variant == "g6_768" else 2,
            QK_PANEL=64 if args.variant == "g6_768" else 128,
            SCORE_TILE=768 if args.variant == "g6_768" else 1024,
            PREFETCH=0 if args.variant == "g6_768" else 1,
            TO_L0=True,
            BALANCED=False,
            ROOTS=2,
            OA_COLS=128,
            OB_ROWS=96,
            OUTPUT_GROUP_PIPELINE=True,
            HC_ROW_DEPENDENCIES=True,
            ATTENTION_QUERY_METADATA=False,
            HC_CONSUMER_REDUCE=False,
        )
        report["controls"] = controls
        roots = {}
        for label, source in [("reference", REFERENCE), ("candidate", CANDIDATE)]:
            package = load_package(source, args.variant + "_" + label)
            kernel = importlib.import_module(package.name + ".entry").decode_csa_spmd
            artifact = kernel._resolve_kernel_artifact(
                (),
                {
                    **controls,
                    "config": CompileOptions(
                        platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO, output_dir=str(args.output / label)
                    ),
                },
                allow_signature_mode=True,
            )
            root = Path(artifact.output_dir)
            roots[label] = root
            report[label] = dict(artifact=str(root), full_codegen_passed=True)
            audit_codegen(root, report[label])
            pto = (root / "ptoas/dynamic_csa_full.pto").read_text()
            report[label]["no_local_half_buffer"] = "buf_half_scores" not in pto
            report[label]["no_local_half_math"] = "ub_half_" not in pto
            assert report[label]["no_local_half_buffer"] and report[label]["no_local_half_math"]
        comparisons = []
        for left in sorted((roots["reference"] / "kernels").rglob("*.cpp")):
            right = roots["candidate"] / left.relative_to(roots["reference"])
            comparisons.append(
                dict(
                    kernel=str(left.relative_to(roots["reference"])),
                    equal_without_comments_whitespace=normalize_cpp(left.read_text())
                    == normalize_cpp(right.read_text()),
                )
            )
        report["kernel_comparison"] = comparisons
        report["codegen_identical"] = all(row["equal_without_comments_whitespace"] for row in comparisons)
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
