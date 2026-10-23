# SPDX-License-Identifier: Apache-2.0
"""CPU阶段先解析两种内部工作区入口的完整依赖图，不占卡。"""

import argparse
import importlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--specialize", action="store_true", help="在CPU完成完整前端特化及IR解析")
parser.add_argument("--compile-output", type=Path, help="CPU完成完整编译流水线并保存产物，不执行设备")
parser.add_argument("--entry", help="仅检查指定入口，用于定位编译问题")
parser.add_argument("--kernel-artifact", action="store_true", help="按实际eager的kernel ABI生成产物，不运行设备")
parser.add_argument("--service-binding", action="store_true", help="编译实际服务entry.py而不是私有原型模块")
parser.add_argument("--shape", type=int, nargs=2, metavar=("TOKENS", "REQUESTS"), help="只检查指定的已知容量")
args = parser.parse_args()
# 与单卡/整模型的固定NZ2合同一致，须在导入算子布局常量之前设置。
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

for package, name in (
    ("deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd_internal_debug"),
    ("deepseek_v4_flash_csa_tnd", "decode_csa_tnd_internal_debug"),
):
    if args.service_binding:
        name = name.replace("_internal_debug", "_debug")
    source = "entry" if args.service_binding else "internal_prototype"
    module = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.{source}")
    for entry in (name, name.removesuffix("_debug")):
        if args.entry and args.entry != entry:
            continue
        graph = getattr(module, entry)._get_dep_graph()
        print(
            json.dumps({"package": package, "entry": entry, "status": "PARSED", "graph_type": type(graph).__name__}),
            flush=True,
        )
        if args.specialize or args.compile_output:
            select = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.policy").select_policy
            for tokens, requests in [args.shape] if args.shape else ((16, 16), (144, 24)):
                policy = select(
                    tokens=tokens,
                    requests=requests,
                    max_query_tokens=6,
                    max_seq_len=131078,
                    compressed_table_columns=2048,
                )
                program = getattr(module, entry).specialize(**policy.constexprs())
                from pypto.ir._kernel_compile import kernel_signature_for_program

                params, aliases = kernel_signature_for_program(program)
                print(
                    json.dumps(
                        {
                            "entry": entry,
                            "tokens": tokens,
                            "requests": requests,
                            "status": "SPECIALIZED",
                            "program_type": type(program).__name__,
                            "kernel_external_parameters": len(params),
                            "kernel_return_aliases": list(aliases),
                        }
                    ),
                    flush=True,
                )
                if args.compile_output:
                    from pypto import passes
                    from pypto.runtime import CompileOptions

                    output = args.compile_output.resolve() / f"{entry}_T{tokens}_B{requests}"
                    if args.kernel_artifact:
                        from pypto.ir._kernel_compile import kernel_abi_for_program
                        from pypto.ir.compile import _compile_impl

                        abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer")
                        _compile_impl(
                            program,
                            _kernel_abi=abi,
                            platform="a2a3",
                            output_dir=str(output),
                            memory_planner=passes.MemoryPlanner.PYPTO,
                            dump_passes=False,
                        )
                    else:
                        getattr(module, entry).compile(
                            **policy.constexprs(),
                            config=CompileOptions(
                                platform="a2a3", output_dir=str(output), memory_planner=passes.MemoryPlanner.PYPTO
                            ),
                        )
                    print(
                        json.dumps(
                            {
                                "entry": entry,
                                "tokens": tokens,
                                "requests": requests,
                                "status": "COMPILED",
                                "compile_mode": "kernel" if args.kernel_artifact else "program",
                                "output": str(output),
                            }
                        ),
                        flush=True,
                    )
