# SPDX-License-Identifier: Apache-2.0
"""用公共工具链编译TND根入口，记录真实签名与选定业务档位。"""

import argparse
import importlib
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("spmd", "mpmd"), required=True)
    parser.add_argument("--case", default="h131072_b16_s6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device_execution": False, "variant": args.variant}
    try:
        from load_snapshot import load_snapshot
        from pypto import passes
        from pypto.runtime import CompileOptions

        here = Path(__file__).resolve().parent
        cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
        case = next(c for c in cases if c["name"] == args.case)
        suffix, symbol = ("spmd_tnd", "decode_csa_spmd") if args.variant == "spmd" else ("tnd", "decode_csa_tnd")
        namespace = load_snapshot(args.snapshot, "compile") if args.snapshot else "vllm_ascend.ops.pypto"
        package = namespace + ".deepseek_v4_flash_csa_" + suffix
        layout_module = importlib.import_module(namespace + ".deepseek_v4_flash_csa.nz_mode")
        policy = importlib.import_module(package + ".policy").select_policy(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        )
        module = importlib.import_module(package + ".entry")
        entry = getattr(module, symbol)
        layouts, shapes = layout_module.root_weight_layouts(entry), layout_module.root_weight_shapes(entry)
        for name in ("wkv", "idx_wq_b", "weights_proj"):
            assert layouts[name] == "NZ", (name, layouts[name])
        assert shapes["wkv"] == (512, 4096)
        assert shapes["idx_wq_b"] == (1024, 8192)
        assert shapes["weights_proj"] == (64, 4096)
        report.update(case=case, policy=policy.constexprs(), source=module.__file__, layouts=layouts, shapes=shapes)
        entry.compile(
            **policy.constexprs(),
            config=CompileOptions(
                platform="a2a3", output_dir=str(args.output / "kernel"), memory_planner=passes.MemoryPlanner.PYPTO
            ),
        )
        report["passed"] = True
    except Exception:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
