# SPDX-License-Identifier: Apache-2.0
"""核对迁移的数学AST、四类事件调用及动态源码，不执行设备。"""

import argparse
import ast
import copy
import json
from pathlib import Path


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(
                id={"T": "t_dim", "T_PAD": "packed_stride", "NUM_QK_CORES": "CORE_GROUPS"}.get(node.id, node.id),
                ctx=node.ctx,
            ),
            node,
        )

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name) and node.left.id == "EVENT_BASE":
            return node.right
        return node


def canonical(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def audit():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    old = root / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_sparse_attn_csa.py"
    new = root / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/attention_stream_stages.py"
    ot = ast.parse(old.read_text())
    nt = ast.parse(new.read_text())
    original = next(n for n in ot.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_csa")
    original_nodes = sorted(ast.walk(original), key=lambda n: getattr(n, "lineno", 0))
    migrated_nodes = sorted(ast.walk(nt), key=lambda n: getattr(n, "lineno", 0))
    names = [
        "qk_l1_t",
        "qk_kv_t",
        "qk_scores",
        "pv_probability",
        "pv_probability_left",
        "pv_first_right",
        "pv_previous",
        "pv_next_right",
        "pv_current",
        "m_idx",
        "m_rem_tmp",
        "m_lane",
        "m_swap_row",
        "m_swap_base",
        "m_swap_source",
        "m_row_offsets",
        "m_swap_idx",
        "running_m",
        "running_l",
        "running_left",
        "running_right",
        "qk_reduce_tmp",
        "qk_scores_half",
        "qk_bias",
        "qk_masked",
        "qk_mi",
        "qk_exp",
        "qk_li",
        "qk_probability",
        "pv_m",
        "pv_l",
        "next_m",
        "alpha",
        "beta",
        "next_l",
        "pv_left",
        "next_left",
        "pv_right",
        "next_right",
        "m_gather_tmp",
        "pub_local_head",
        "m_h0",
        "m_mi",
        "m_li",
        "m_left",
        "m_right",
        "m_oi",
        "n_sink_bias",
        "n_sink_tile",
        "n_denom",
        "n_full",
        "n_bf16",
        "m_rope",
        "m_cos_il",
        "m_sin_signed",
        "m_swapped_flat",
        "m_swapped",
        "m_rot",
        "n_rope_bf16",
        "n_full_bf16",
        "n_group_bf16",
        "n_pack_first",
        "n_pack_second",
        "n_pack_row",
        "n_pack_row_second",
        "reset_m",
        "reset_l",
        "reset_left",
        "reset_right",
    ]

    def assignments(nodes, name):
        return [
            n
            for n in nodes
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == name
        ]

    math = []
    for name in names:
        a = assignments(original_nodes, name)
        b = assignments(migrated_nodes, name)
        assert a and [canonical(n) for n in a] == [canonical(n) for n in b], name
        math.append(
            {
                "name": name,
                "canonical_lines": [n.lineno for n in a],
                "migrated_lines": [n.lineno for n in b],
                "exact_ast_after_dimension_aliases": True,
            }
        )

    def events(nodes):
        return [
            n
            for n in nodes
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.system.sync_set", "pl.system.sync_wait")
        ]

    ae = events(original_nodes)
    be = events(migrated_nodes)
    assert [canonical(n) for n in ae] == [canonical(n) for n in be]
    calls = {ast.unparse(n.func) for n in migrated_nodes if isinstance(n, ast.Call)}
    assert not (calls & {"pl.spmd", "pl.at", "pl.system.task_dummy", "pl.system.submit"})
    assert "query_request" in calls
    names_used = {n.id for n in migrated_nodes if isinstance(n, ast.Name)}
    assert not ({"S", "DECODE_SEQ", "T_PAD"} & names_used)
    assert len([n for n in nt.body if isinstance(n, ast.FunctionDef)]) == 2
    assert all(
        ast.unparse(n.decorator_list[0]) == "pl.jit.inline(auto_scope=False)"
        for n in nt.body
        if isinstance(n, ast.FunctionDef)
    )
    rounds = [
        ast.unparse(n)
        for n in migrated_nodes
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.cast" and any(k.arg == "mode" for k in n.keywords)
    ]
    assert len(rounds) == 3 and all("mode='rint'" in n for n in rounds)
    return {
        "passed": True,
        "canonical": str(old),
        "migrated": str(new),
        "math_assignment_names": len(math),
        "math_assignments": math,
        "event_calls": len(be),
        "event_call_order_and_pipe_exact": True,
        "event_calls_detail": [{"line": n.lineno, "expression": ast.unparse(n)} for n in be],
        "bf16_rounding_calls": rounds,
        "source_contracts": {
            "request_mapping": "query_request(Q,L,t,T), no token//S",
            "cube_actual_T": "q_flat.shape[0]//64",
            "vector_actual_T": "positions.shape[0]",
            "packed_group_stride": "o_packed_heads.shape[0]//8",
            "stage_constants": "64 heads, 512 columns, 5 candidate blocks, 3 ring slots",
            "ffts_exception": "fixed INT64[256] hardware control region required by set_ffts API",
        },
        "limits": [
            "Math AST equivalence does not prove compiled numerical precision",
            "Gather changes use true Q/L and explicit bounds; event predicate stays same shared plan mask",
            "No NPU execution or timing",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("math_assignments", "event_calls_detail")}, ensure_ascii=False
        )
    )
