# SPDX-License-Identifier: Apache-2.0
"""保留独立Plan task，仅迁移来源八行tile/48路分工，去掉无任务worker。"""

import argparse
import ast
import json
import shutil

from prepare import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="plan_workers_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "decode_sparse_attn_csa.py"
    source = path.read_text()
    original = "CSA_PLAN_WORKERS = 16  # csa_slots_build_valid_qk_plan token-tile lanes"
    if source.count(original) != 1:
        raise ValueError("来源Plan worker上限不符")
    source = source.replace(
        original,
        "CSA_PLAN_WORKERS = 48  # 冻结单SPMD的八行Plan条带；独立task只派发非空worker",
        1,
    )
    original = 'with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True)'
    if source.count(original) != 1:
        raise ValueError("来源Plan任务入口不唯一")
    source = source.replace(
        original,
        "with pl.spmd(pl.min(CSA_PLAN_WORKERS, (t_dim + BIAS_T_TILE - 1) // BIAS_T_TILE),\n"
        '                 name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True)',
        1,
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="冻结13259f92：attention_slot_plan_stage八行tile按48个AIV轮转",
                scope="Plan仍独立task；launch=min(48,ceil(T/8))，固定48路stride覆盖多wave",
                preserved="Plan数学、尾行、页保护、QK/PV及全部数据依赖不变",
                lifetime="没有新增GM信号、软等待或scratch",
                status="source_prepared_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
