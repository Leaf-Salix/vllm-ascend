# SPDX-License-Identifier: Apache-2.0
"""使用公共bootstrap编译CSA入口，不提供样例张量、不执行设备。"""

import argparse
import importlib
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("bsh", "spmd_tnd", "mpmd_tnd"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device_execution": False, "variant": args.variant}
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        from pypto import passes
        from pypto.runtime import CompileOptions

        suffix, symbol = {
            "bsh": ("spmd", "decode_csa_spmd"),
            "spmd_tnd": ("spmd_tnd", "decode_csa_tnd"),
            "mpmd_tnd": ("tnd", "decode_csa_tnd"),
        }[args.variant]
        package = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_" + suffix
        policy = importlib.import_module(package + ".policy").select_policy(
            tokens=args.tokens, requests=args.requests, max_query_tokens=6, max_seq_len=args.max_seq_len
        )
        entry = importlib.import_module(package + ".entry")
        report.update(source=entry.__file__, policy=policy.constexprs())
        getattr(entry, symbol).compile(
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
