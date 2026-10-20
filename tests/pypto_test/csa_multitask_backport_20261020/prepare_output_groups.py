# SPDX-License-Identifier: Apache-2.0
"""适配13259f92已验证的输出组流水为多task的局部MIX尾段。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_scale_early import add_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="output_groups_v1")
    parser.add_argument("--original-ob-cache", action="store_true", help="保留原多task NZ O_B默认缓存属性")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    for name in ("output_projection_a_stages.py", "output_projection_b_stages.py", "phase_sync.py", "signals.py"):
        source = (HERE / "sources/spmd" / name).read_text()
        source = source.replace("from ..deepseek_v4_flash_csa.config import", "from .config import")
        if name == "phase_sync.py":
            # 局部MIX的arena容量由固定17阶段/72角色决定；保持同一固定描述符
            # 穿越条件，避免helper返回时将分支内绑定的dynamic容量带出作用域。
            source = source.replace(
                'SIGNAL_WORDS = pl.dynamic("CSA_PHASE_SIGNAL_WORDS")',
                "SIGNAL_WORDS = (2 * 8 + 1) * (3 * 24) * 128",
                1,
            )
        if name.startswith("output_projection_"):
            # 多task现有workspace是固定T_PAD容量，实际T仍由业务Tensor动态轴传入。
            # 将helper容量合同绑定到这份真实容量，避免固定GM形状中残留无法物化的
            # 单SPMD独立dynamic符号；不改变任何数学循环的实际tokens边界。
            source = source.replace(
                "import pypto.language as pl\n",
                "import pypto.language as pl\n\nfrom .decode_o_proj import T_PAD\n",
                1,
            )
            for axis in ("MM_ROWS_DYN", "SCALE_ROWS_DYN"):
                start = source.index(axis + " = pl.dynamic(")
                end = source.index("\n", start)
                source = source[:start] + axis + " = T_PAD" + source[end:]
            if name == "output_projection_a_stages.py":
                source = source.replace(
                    'PACK_TOTAL_ROWS_DYN = pl.dynamic("CSA_OA_PACK_TOTAL_ROWS")',
                    "PACK_TOTAL_ROWS_DYN = O_GROUPS * T_PAD",
                    1,
                )
        (target / name).write_text(source)
    source = (HERE / "output_group_mixed_template.py").read_text()
    if args.original_ob_cache:
        marker = 'pl.func_attr({"cache_policy": [(1, 1), (2, 1)]})'
        if source.count(marker) != 1:
            raise AssertionError("需要唯一的输出Cube缓存属性")
        source = source.replace(marker, 'pl.func_attr({"cache_policy": [(1, 1)]})', 1)
    (target / "output_group_mixed.py").write_text(source)
    path = target / "decode_csa.py"
    source = path.read_text()
    source = source.replace(
        "from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, decode_o_proj_tp1\n",
        "from .decode_o_proj import LOCAL_T, LOCAL_T_PAD\n"
        "from .output_group_mixed import OUTPUT_SIGNAL_WORDS, decode_output_groups, output_flags_init\n",
        1,
    )
    marker = "        x_hc32 = pl.create_tensor([t_dim, HC_MULT, D], dtype=pl.FP32)\n"
    if source.count(marker) != 1:
        raise AssertionError("需要唯一的前段workspace分配点")
    source = source.replace(
        marker,
        "        output_flags = pl.create_tensor([OUTPUT_SIGNAL_WORDS], dtype=pl.INT32)\n"
        "        output_flags, output_init_tid = output_flags_init(output_flags)\n"
        "        output_init_dependencies = pl.array.create(1, pl.TASK_ID)\n"
        "        output_init_dependencies[0] = output_init_tid\n" + marker,
        1,
    )
    source = source.replace("decode_o_proj_tp1(", "decode_output_groups(")
    source = add_calls(source, ("decode_output_groups",), ("output_flags", "output_init_dependencies[0]"))
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="2417166f，冻结13259f92的OUTPUT_GROUP_PIPELINE",
                original_ob_cache=args.original_ob_cache,
                scope="24组局部MIX执行O_A/组量化/O_B/HCpost；完整CSA保持多task",
                preserved="组内K/量化/八组累加/BF16边界；原实际T策略和NZ物理权重",
                lifetime="每次调用独占运行时arena flags，早期显式清零；尾段等待init和packed",
                wait_policy={"backoff": "cached100", "unit_family": "output"},
                limits="私有实验仅CANN9.2/NZ2；未证明设备结果或性能",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
