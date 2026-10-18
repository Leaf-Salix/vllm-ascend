#!/usr/bin/env python3
"""只读核对主会话AIV三个helper；仅标准库AST，不导入或编译PyPTO。"""

import ast
import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
VECTOR = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_vector_step_source/source_candidate/score_vector_steps_g6.py")


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = {"G6_QUERY_GROUP": "query_group_size", "G6_SCORE_TILE": "score_tile"}.get(node.id, node.id)
        return node


def body_dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def main():
    reference = ast.parse((HERE / "source_reference/indexer_score_stages.py").read_text())
    stage = next(n for n in reference.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_vector_lane_stage")
    branch = next(n for n in ast.walk(stage) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                  and n.test.id == "balance_leaves")
    init = [n for n in branch.body if isinstance(n, ast.Assign)
            and n.targets[0].id in ("single_roots", "single_segments")]
    assert len(init) == 2
    step = next(n for n in branch.body if isinstance(n, ast.For) and n.target.id == "single_step")
    publish = next(n for n in branch.body if isinstance(n, ast.For) and n.target.id == "single_local")
    tree = ast.parse(VECTOR.read_text())
    helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    portions = {"score_vector_leaf_state_g6": init, "score_vector_step_g6": step.body,
                "score_vector_leaf_publish_g6": [publish]}
    checks = {}
    for name, expected in portions.items():
        fn = helpers[name]
        assert isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
        assert isinstance(fn.body[-1], ast.Return)
        normalized = Normalize().visit(ast.Module(body=copy.deepcopy(fn.body[1:-1]), type_ignores=[])).body
        assert body_dump(normalized) == body_dump(expected), name
        checks[name + "_complete_body_ast_equal"] = True
    for name in ("score_vector_leaf_state_g6", "score_vector_step_g6"):
        assert ast.unparse(helpers[name].body[-1].value) == "(single_roots, single_segments)"
    assert ast.unparse(helpers["score_vector_leaf_publish_g6"].body[-1].value) == "pair_arena"
    fn = helpers["score_vector_step_g6"]
    top_calls = [(i, ast.unparse(n.value.func)) for i, n in enumerate(fn.body)
                 if isinstance(n, (ast.Assign, ast.Expr)) and isinstance(n.value, ast.Call)]
    wait = next(i for i, name in top_calls if name == "pl.system.sync_wait")
    load = next(i for i, name in top_calls if name == "pl.load")
    ack = next(i for i, name in top_calls if name == "pl.system.sync_set")
    first_query_math = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For)
                            and isinstance(n.target, ast.Name) and n.target.id == "single_local")
    assert wait < load < ack < first_query_math
    for name in ("score_vector_leaf_state_g6", "score_vector_leaf_publish_g6"):
        assert not any(isinstance(n, ast.Call) and ast.unparse(n.func).startswith("pl.system.sync_")
                       for n in ast.walk(helpers[name]))
    globals_ = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    globals_.update(a.asname or a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names)
    for name, fn in helpers.items():
        module = ast.Module(body=fn.body, type_ignores=[])
        loaded = {n.id for n in ast.walk(module) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assigned = {n.id for n in ast.walk(module) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        assert not (loaded - assigned - globals_ - {a.arg for a in fn.args.args}), name
    report = dict(
        passed=True, source=str(VECTOR), source_bytes=VECTOR.stat().st_size, **checks,
        explicit_state_returns=True, unresolved_dynamic_dependencies=[],
        ready_load_ack_top_level_and_before_query_math=True,
        empty_query_lane_still_consumes_and_acks=True,
        leaf_root_store_unchanged=True,
        root_phase_release_remains_callers_responsibility=True,
        limits="仅源码接口/AST；未审计调用者状态机、实际Tile分配或CCEC/设备行为",
        pypto_imported=False, compilation_performed=False, device_execution=False,
    )
    (HERE / "vector_interface_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
