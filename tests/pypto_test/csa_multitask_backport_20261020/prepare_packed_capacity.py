# SPDX-License-Identifier: Apache-2.0
"""迁移已验证单SPMD的pad128 packed容量及组步长，保持多task数学和依赖。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="packed_capacity_v2")
    parser.add_argument("--keep-symbolic-returns", action="store_true", help="复现v1跨inline返回注解的NameError")
    parser.add_argument("--long-single-block-only", action="store_true", help="仅长历史且T<=128缩容，其余保持原容量")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / name for name in (args.base, args.candidate))
    donor = (HERE / "sources/spmd_a241fa61/binding.py").read_text()
    if '"packed": ((8 * p128, 4096), torch.bfloat16)' not in donor:
        raise ValueError("来源必须仍为按实际T对齐128的八组packed")
    changed = {}
    name = "decode_csa.py"
    source = (base / name).read_text()
    source = replace_once(
        source,
        "from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, decode_o_proj_tp1",
        "from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, PACKED_ROW_ALIGN, decode_o_proj_tp1",
    )
    capacity_plan = (
        "        # 来源单SPMD按实际T预留pad128容量；计算放在scope外，保持合并分配。\n"
        "        packed_row_stride = (t_dim + PACKED_ROW_ALIGN - 1) // PACKED_ROW_ALIGN * PACKED_ROW_ALIGN\n"
    )
    if args.long_single_block_only:
        capacity_plan = (
            "        # 仅保留长历史单个128行容量域；短档和多块域继续使用原384行。\n"
            "        packed_row_stride = pl.cast(T_PAD, pl.INDEX)\n"
            "        if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
            "            if t_dim <= PACKED_ROW_ALIGN:\n"
            "                packed_row_stride = pl.cast(PACKED_ROW_ALIGN, pl.INDEX)\n"
        )
    source = replace_once(
        source,
        "        position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])\n",
        "        position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])\n" + capacity_plan,
    )
    source = replace_once(
        source,
        "o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)",
        "o_packed_heads = pl.create_tensor([O_GROUPS * packed_row_stride, O_GROUP_IN], dtype=pl.BF16)",
    )
    changed[name] = source
    for name in ("decode_sparse_attn_csa.py", "decode_o_proj.py"):
        source = (base / name).read_text()
        marker = 'T_DYN = pl.dynamic("T_DYN")'
        source = replace_once(
            source,
            marker,
            'PACKED_ROWS_DYN = pl.dynamic("CSA_PACKED_ROWS_DYN")\n' + marker,
        )
        shape = "pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]"
        if source.count(shape) != 4:
            raise ValueError(f"{name}的四个packed输入/返回类型需要重新核对")
        source = source.replace(shape, "pl.Tensor[[PACKED_ROWS_DYN, O_GROUP_IN], pl.BF16]")
        if name == "decode_sparse_attn_csa.py":
            annotation = ") -> tuple[pl.Tensor[[PACKED_ROWS_DYN, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:"
            if source.count(annotation) != 2:
                raise ValueError("只处理两个Attention返回注解")
            if not args.keep_symbolic_returns:
                # 返回Tensor沿实参保留真实容量表达式，不让helper局部维度名逃逸到根namespace。
                source = source.replace(annotation, "):")
            source = replace_once(
                source,
                "    t_heads = t_dim * H\n",
                "    t_heads = t_dim * H\n    packed_row_stride = pl.tensor.dim(o_packed_heads, 0) // O_GROUPS\n",
            )
            source = replace_once(
                source,
                "n_pack_row = (m_h0 // HEADS_PER_GROUP) * T_PAD + pv_t",
                "n_pack_row = (m_h0 // HEADS_PER_GROUP) * packed_row_stride + pv_t",
            )
            source = replace_once(
                source, "n_pack_row_second = n_pack_row + T_PAD", "n_pack_row_second = n_pack_row + packed_row_stride"
            )
        else:
            source = replace_once(source, marker, "PACKED_ROW_ALIGN = 128\n" + marker)
            source = replace_once(
                source,
                "    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)\n",
                "    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)\n"
                "    packed_row_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS\n",
            )
            source = replace_once(source, "row_base_o = g * T_PAD", "row_base_o = g * packed_row_stride")
        changed[name] = source
    for source in changed.values():
        ast.parse(source)
    shutil.copytree(base, target)
    for name, source in changed.items():
        (target / name).write_text(source)
    manifest = {
        "candidate": args.candidate,
        "base": args.base,
        "changed_files": list(changed),
        "donor": "a241fa61/binding.py:packed + attention_stream_stages.py + output_projection_a_stages.py",
        "change": "packed由固定8*T_PAD改为8*pad128(actual T)，Attention发布和O-A读取共同使用描述符给出的组步长",
        "preserved": "所有token/长度/页表动态有效范围、O-A有效行、任务划分/显式依赖、规约及BF16舍入不变",
        "capacity_only": "其它投影暂存仍保持原T_PAD，不把packed缩容扩大为未验证的全部工作区改造",
        "resource_rule": "按实际T对齐128，无batch/seq_len测试点白名单；各组容量始终不少于实际T",
        "symbolic_returns": args.keep_symbolic_returns,
        "long_single_block_only": args.long_single_block_only,
        "status": "private_candidate_no_device_evidence",
    }
    if args.long_single_block_only:
        manifest["change"] = "仅长历史且实际T<=128时采用8×128 packed；其它区间保持8×T_PAD，发布/读取共用动态组步长"
        manifest["resource_rule"] = "压缩历史>8192且T<=128属于一个128行容量块；无batch白名单，其余保留原容量"
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
