# SPDX-License-Identifier: Apache-2.0
"""只验证META置换/逆置换与边界保护；不把此微例当作完整CSA或性能证明。"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import pypto
import pypto.language as pl

from score_owner_model import base, expected, fixtures, pad16

HERE = Path(__file__).resolve().parent
LOADER = HERE.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/mgather_physical_plan_retest_20261019"
sys.path.insert(0, str(LOADER))
from source_loader import load_package

SOURCE_PARSER = argparse.ArgumentParser(add_help=False)
SOURCE_PARSER.add_argument("--spmd-source", type=Path, default=HERE / "source_score_owner_balance")
SOURCE_OPTIONS, _ = SOURCE_PARSER.parse_known_args()
PACKAGE = load_package(SOURCE_OPTIONS.spmd_source, "score_owner_probe")
PLAN = importlib.import_module(PACKAGE.name + ".score_owner_stages").score_balance_plan_stage
WORDS = pl.dynamic("SCORE_OWNER_PROBE_WORDS")
REQUESTS = pl.dynamic("SCORE_OWNER_PROBE_REQUESTS")
TOKENS = pl.dynamic("SCORE_OWNER_PROBE_TOKENS")


@pl.jit.incore
def core(workspace: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
         plan: pl.Tensor[[8], pl.INT32], lengths: pl.Tensor[[REQUESTS], pl.INT32],
         positions: pl.Tensor[[TOKENS], pl.INT64]):
    requests = pl.tensor.dim(lengths, 0)
    tokens = pl.tensor.dim(positions, 0)
    workspace = PLAN(workspace, plan, requests, tokens)
    pl.system.bar_all()
    pl.system.cacheinvalid()
    pl.system.fence()
    return workspace


@pl.jit
def entry(workspace: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
          plan: pl.Tensor[[8], pl.INT32], lengths: pl.Tensor[[REQUESTS], pl.INT32],
          positions: pl.Tensor[[TOKENS], pl.INT64]):
    with pl.spmd(1):
        workspace = core(workspace, plan, lengths, positions)
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--spmd-source", type=Path, default=HERE / "source_score_owner_balance")
    parser.add_argument("--uniform-cases", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, full_csa=False, performance_measured=False, cases=[])
    try:
        import torch
        import torch_npu  # noqa: F401
        from pypto import torch as pypto_torch
        from pypto.runtime import CompileOptions
        from pypto import passes

        device = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device
        torch.npu.set_device(device)
        pypto_torch.init(device=device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        pypto.configure_cache(pypto.CacheConfig(enabled=False))
        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "compiled"),
                                memory_planner=passes.MemoryPlanner.PYPTO)
        cases = fixtures()
        if args.uniform_cases:
            for name, groups, leaves, extra, visible in (
                ("uniform48_coprime", 48, 5, 0, [32769] * 48),
                ("uniform24_leaf7_extra", 24, 7, 3, [34817] * 24),
                ("uniform24_leaf1", 24, 1, 0, [8193] * 24),
                ("noncoprime_leaf4", 24, 4, 0, [24577] * 24),
                ("nonuniform24_fallback", 24, 5, 0, [32769 if i % 2 else 8193 for i in range(24)]),
                ("uniform24_negative_visible", 24, 5, 0, [-1] * 24),
            ):
                cases.append(dict(name=name, requests=groups, tokens=192,
                                  visible=visible, plan=[groups, max(visible), leaves, 8, extra, 6, -1, 1]))
        for case in cases:
            start = base(case["requests"], case["tokens"])
            host = torch.full((start + 96 + 16,), 123456, dtype=torch.int32)
            for group, value in enumerate(case["visible"]):
                host[pad16(case["requests"] + 1) + group * 4 + 3] = value
            workspace = host.to(f"npu:{device}")
            plan = torch.tensor(case["plan"], dtype=torch.int32, device=workspace.device)
            lengths = torch.zeros(case["requests"], dtype=torch.int32, device=workspace.device)
            positions = torch.zeros(case["tokens"], dtype=torch.int64, device=workspace.device)
            entry(workspace, plan, lengths, positions, config=config)
            torch.npu.synchronize()
            result = workspace.cpu()
            want = expected(case["plan"], case["visible"], case["requests"], case["tokens"])
            got = result[start:start + 96].tolist()
            forward, inverse = got[48:72], got[72:96]
            row = dict(name=case["name"], exact=got == want,
                       bijection=sorted(forward) == list(range(24)),
                       inverse=all(0 <= forward[i] < 24 and inverse[forward[i]] == i for i in range(24)),
                       prefix_unchanged=torch.equal(result[:start], host[:start]),
                       suffix_unchanged=torch.equal(result[start + 96:], host[start + 96:]),
                       words=got)
            row["passed"] = all(row[key] for key in ("exact", "bijection", "inverse", "prefix_unchanged", "suffix_unchanged"))
            report["cases"].append(row)
            if not row["passed"]:
                raise AssertionError(case["name"])
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]), "error": report.get("error", "")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
