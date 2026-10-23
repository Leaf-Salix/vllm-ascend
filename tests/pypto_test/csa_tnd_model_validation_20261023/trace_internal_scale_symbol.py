# SPDX-License-Identifier: Apache-2.0
"""只记录scale符号相关签名和IR行；不生成全量pass dump，不执行设备。"""

import argparse
import json
import os
from pathlib import Path

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

from pypto import ir
from pypto.ir.compile import _run_pass_pipeline
from pypto.pypto_core import passes
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.internal_prototype import decode_csa_spmd_internal_debug

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
report = []


def record(label, program):
    for function in program.functions.values():
        if "vector_schedule" not in function.name and "dynamic_csa_full" not in function.name:
            continue
        params = [
            (value.name_hint, ir.python_print_type(value.type))
            for value in function.params
            if "o_scales" in value.name_hint or "O_SCALE_ROWS" in ir.python_print_type(value.type)
        ]
        code = ir.python_print(function)
        lines = [(i + 1, line[:900]) for i, line in enumerate(code.splitlines()) if "CSA_FULL_O_SCALE_ROWS" in line]
        report.append(
            {
                "pass": label,
                "function": function.name,
                "params": params,
                "matching_lines": lines[:24],
                "line_count": len(lines),
            }
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def after_pass(pass_, program):
    if pass_.get_name() in {
        "InlineFunctions",
        "ConvertToSSA",
        "ExpandMixedKernel",
        "LowerPipelineLoops",
        "MaterializeTensorStrides",
        "MaterializeValidShapeSymbols",
    }:
        record(pass_.get_name(), program)


policy = select_policy(tokens=16, requests=16, max_query_tokens=6, max_seq_len=131078, compressed_table_columns=2048)
program = decode_csa_spmd_internal_debug.specialize(**policy.constexprs())
record("specialize", program)
_run_pass_pipeline(
    program,
    operation="compile",
    platform="a2a3",
    memory_planner=passes.MemoryPlanner.PYPTO,
    extra_instruments=(passes.CallbackInstrument(after_pass=after_pass),),
)
print("Saved targeted scale-symbol trace:", args.output)
