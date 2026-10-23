# SPDX-License-Identifier: Apache-2.0
"""CPU阶段先解析两种内部工作区入口的完整依赖图，不占卡。"""

import argparse
import importlib
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--specialize", action="store_true", help="在CPU完成完整前端特化及IR解析")
args = parser.parse_args()

for package, name in (
    ("deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd_internal_debug"),
    ("deepseek_v4_flash_csa_tnd", "decode_csa_tnd_internal_debug"),
):
    module = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.internal_prototype")
    for entry in (name, name.removesuffix("_debug")):
        graph = getattr(module, entry)._get_dep_graph()
        print(
            json.dumps({"package": package, "entry": entry, "status": "PARSED", "graph_type": type(graph).__name__}),
            flush=True,
        )
        if args.specialize:
            select = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.policy").select_policy
            for tokens, requests in ((16, 16), (144, 24)):
                policy = select(
                    tokens=tokens,
                    requests=requests,
                    max_query_tokens=6,
                    max_seq_len=131078,
                    compressed_table_columns=2048,
                )
                program = getattr(module, entry).specialize(**policy.constexprs())
                print(
                    json.dumps(
                        {
                            "entry": entry,
                            "tokens": tokens,
                            "requests": requests,
                            "status": "SPECIALIZED",
                            "program_type": type(program).__name__,
                        }
                    ),
                    flush=True,
                )
