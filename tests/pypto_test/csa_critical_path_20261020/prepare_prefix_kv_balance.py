# SPDX-License-Identifier: Apache-2.0
"""从采用Score后的正式快照构造KV逻辑worker置换候选。

只调整投影前段末尾的KV任务归属，不改变KV数学、任务总数、Score叶/根
编号或核间协议。辅助Hadamard核先领取KV，其次为Q_B列块较少的核，
最后为多一个Q_B列块的核。所有24个逻辑worker仍恰好出现一次。
"""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import textwrap

import prepare_qb_stream_reuse as wiring

HERE = Path(__file__).resolve().parent
FLAG = "BALANCE_PREFIX_KV"


def main():
    base = HERE / "source_after_score_split"
    target = HERE / "source_prefix_kv_balance"
    target.mkdir(exist_ok=False)
    for p in base.glob("*.py"):
        (target / p.name).write_bytes(p.read_bytes())
    old_flag = wiring.FLAG
    try:
        wiring.FLAG = FLAG
        for name, definitions, calls in (
            ("entry.py", ("_csa_core_group", "decode_csa_spmd"), ("_csa_core_group", "cube_schedule")),
            ("cube_schedule.py", ("cube_schedule",), ("projection_overlap_stage",)),
            ("projection_overlap_stages.py", ("projection_overlap_stage",), ()),
        ):
            path = target / name
            path.write_text(wiring.wire(path.read_text(), definitions, calls))
    finally:
        wiring.FLAG = old_flag
    path = target / "projection_overlap_stages.py"
    source = path.read_text()
    before = "from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage"
    assert source.count(before) == 1
    source = source.replace(before, "from .query_projection_stages import QUANT_WEIGHT_LAYOUT, QPROJ_N_BLOCKS, q_b_stage")
    before = "    kv_worker = (worker + GROUPS - BRANCH_GROUPS) % GROUPS\n"
    assert source.count(before) == 1
    replacement = before + """    if BALANCE_PREFIX_KV:
        # 同一个24核集合的双射，不把KV压到四个辅助核。先利用已完成
        # Hadamard的R核，再用Q_B少一轮的核，最后才用Q_B多一轮的核。
        # Q_B的列任务数来自数学N tile；真实T/行组/尾行仍由原KV helper处理。
        # 原消费者在Score后等待全部24个KV phase，不依赖旧物理owner公式。
        qb_workers = GROUPS - BRANCH_GROUPS
        qb_extra = QPROJ_N_BLOCKS % qb_workers
        if worker < BRANCH_GROUPS:
            kv_worker = worker
        else:
            qb_owner = worker % qb_workers
            if qb_owner >= qb_extra:
                kv_worker = BRANCH_GROUPS + qb_owner - qb_extra
            else:
                kv_worker = GROUPS - qb_extra + qb_owner
"""
    source = source.replace(before, replacement)
    marker = source.index('"""', source.index('"""') + 3)
    source = source[:marker] + """
本轮候选总览
    BALANCE_PREFIX_KV在长历史G6的T>=128资源区间尝试新的KV逻辑worker
    置换：辅助Hadamard核优先，其次为Q_B少一列块的核。128来自KV由M32
    转M64的既有资源边界，不枚举B/S验收点。每核仍调用同一个KV helper，
    按真实device T处理完整K4096及尾行；全部24核各发布一次原KV phase。
    这与历史只用四辅助核执行KV不同：没有缩减KV并行度或移动生产者次序。
    候选尚未采用，真实收益须通过相对新正式Score的同轮完整L1门槛。
""" + source[marker:]
    path.write_text(source)
    path = target / "policy.py"
    source = path.read_text()
    for old, new in (
        ("SCORE_SPLIT_MIN_TOKENS = 32\n", "SCORE_SPLIT_MIN_TOKENS = 32\n# 与KV投影从M32切换M64的资源边界一致。\nKV_BALANCE_MIN_TOKENS = 128\n"),
        ("    split_score_rows: bool = False\n", "    split_score_rows: bool = False\n    balance_prefix_kv: bool = False\n"),
        ('            "SPLIT_SCORE_ROWS": self.split_score_rows,\n', '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n            "BALANCE_PREFIX_KV": self.balance_prefix_kv,\n'),
        ("        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n", """        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,
        # 既有KV投影在128行转M64；仅长G6先评估辅助/少轮Q_B核优先置换。
        # 不改变24核逻辑集合或原device T循环，短历史暂不扩档。
        balance_prefix_kv=long_history and group == 6 and tokens >= KV_BALANCE_MIN_TOKENS,
"""),
    ):
        assert source.count(old) == 1
        source = source.replace(old, new)
    path.write_text(source)

    # 直接执行候选中实际的Scalar置换表达式，不另写一个不同的映射作自证。
    mapping_ast = ast.parse(textwrap.dedent(replacement))
    mapping_code = compile(mapping_ast, "candidate_kv_mapping", "exec")
    ownership = []
    for groups in (24,):
        for branches in range(1, groups):
            for columns in (64, 128):
                mapped = []
                for worker in range(groups):
                    scope = dict(worker=worker, GROUPS=groups, BRANCH_GROUPS=branches,
                                 QPROJ_N_BLOCKS=columns, BALANCE_PREFIX_KV=True)
                    exec(mapping_code, {}, scope)
                    mapped.append(scope["kv_worker"])
                assert sorted(mapped) == list(range(groups))
                ownership.append(dict(groups=groups, branches=branches, columns=columns,
                                      physical_to_logical=mapped))
    spec = importlib.util.spec_from_file_location("prefix_kv_policy", target / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controls = {}
    for batch in (16, 28):
        control = module.select_policy(tokens=batch*6, requests=batch, max_query_tokens=6,
                                       max_seq_len=131078).constexprs()
        controls[str(batch)] = control
        assert control["SPLIT_SCORE_ROWS"] and control[FLAG] == (batch == 28)
        (HERE / f"prefix_kv_controls_b{batch}.json").write_text(json.dumps(control, indent=2) + "\n")
    changed = sorted(p.name for p in target.glob("*.py") if p.read_bytes() != (base/p.name).read_bytes())
    for p in target.glob("*.py"):
        ast.parse(p.read_text())
    report = dict(base="source_after_score_split/13259f92", changed=changed,
                  scalar_permutations_checked=ownership, controls=controls,
                  kv_math_file_unchanged=True, score_math_and_owner_unchanged=True,
                  formal_changed=False, device_validated=False, performance_measured=False)
    (HERE / "PREFIX_KV_BALANCE_PREPARE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print({"changed": changed, "permutations": len(ownership), "controls": controls})


if __name__ == "__main__":
    main()
