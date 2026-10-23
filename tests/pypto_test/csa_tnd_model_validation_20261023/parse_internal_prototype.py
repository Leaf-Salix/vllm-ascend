# SPDX-License-Identifier: Apache-2.0
"""CPU阶段先解析两种内部工作区入口的完整依赖图，不占卡。"""

import importlib
import json

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
