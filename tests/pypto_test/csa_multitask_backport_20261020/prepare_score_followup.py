# SPDX-License-Identifier: Apache-2.0
"""针对协同前段实测启动不均，逐项适配Attention整组启动与KV就绪次序。"""

import argparse
import ast
import json
import shutil

from prepare import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="cooperative_plan_v1")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--attention-sync-start", action="store_true")
    parser.add_argument("--kv-after-qb", action="store_true")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    if not (args.attention_sync_start or args.kv_after_qb):
        raise ValueError("必须选择一个待验证的调度变化")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    changes = []
    if args.attention_sync_start:
        path = target / "decode_sparse_attn_csa.py"
        source = path.read_text()
        before = 'with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, rope_tid], allow_early_resolve=True)'
        if source.count(before) != 1:
            raise ValueError("Attention任务入口不唯一或已经设置sync_start")
        after = before[:-1] + ", sync_start=True)"
        source = source.replace(before, after, 1)
        ast.parse(source)
        path.write_text(source)
        changes.append("仅qk_pv的24路MIX设置sync_start=True，保留原allow_early_resolve")
    if args.kv_after_qb:
        path = target / "decode_csa.py"
        source = path.read_text()
        before = "q_rope_sin_signed, q_rope_swap_idx, kv, score_ready[0],"
        if source.count(before) != 1:
            raise ValueError("KV的Score依赖入口不唯一")
        source = source.replace(before, "q_rope_sin_signed, q_rope_swap_idx, kv, qb_tid,", 1)
        source = source.replace(
            "# 原KV与Q反量化在下方通过直接Score TaskId后置。",
            "# 原KV在QB后具备派发资格；只有Q反量化通过Score TaskId后置。",
            1,
        ).replace(
            "# 优先Score，避免Q/KV尾段抢占其启动窗口；保留allow_early_resolve。",
            "# 仅Q反量化依赖Score；KV沿QB依赖，保留allow_early_resolve。",
            1,
        )
        ast.parse(source)
        path.write_text(source)
        changes.append("KV恢复协同前段的QB依赖；只有Q反量化保留Score依赖")
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="13259f92协同前段的多task依赖适配；不增加新的算术策略",
                changes=changes,
                evidence=(
                    "cooperative_plan_v1长B28五窗中一窗1364.32us，"
                    "QK24路在物理AIC5执行两块、AIC2零块，导致209.92us首启离散"
                ),
                preserved="QK/PV及前段数学、真实输入依赖、动态T与页保护、既有early resolve",
                scope="针对已观察的调度现象逐项验证，不预设sync_start必然加速",
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
