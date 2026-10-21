# SPDX-License-Identifier: Apache-2.0
"""在新冻结包中组合已单测过的Indexer优先和同owner Vector融合。"""

import ast
import copy
import subprocess
import sys
import textwrap
from pathlib import Path


def scope_name(node):
    if not isinstance(node, ast.With) or len(node.items) != 1:
        return None
    call = node.items[0].context_expr
    if not isinstance(call, ast.Call) or ast.unparse(call.func) != "pl.spmd":
        return None
    return next((item.value.value for item in call.keywords if item.arg == "name_hint"), None)


def set_keyword(scope, key, value):
    call = scope.items[0].context_expr
    call.keywords = [item for item in call.keywords if item.arg != key]
    call.keywords.append(ast.keyword(arg=key, value=value))


class OwnerIndex(ast.NodeTransformer):
    def visit_Call(self, node):
        if ast.unparse(node.func) == "pl.tile.get_block_idx":
            return ast.Name(id="owner_worker", ctx=ast.Load())
        return self.generic_visit(node)


def main():
    here = Path(__file__).resolve().parent
    name = sys.argv[1]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    path = here / "experiments" / name / "operator/entry.py"
    source = path.read_text()
    root = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_tnd"
    )
    scopes = {scope_name(node): node for node in ast.walk(root) if scope_name(node)}
    replacements = {}

    for first, second, combined in (
        ("hc_pre_gates", "hc_mix_norm", "hc_gate_mix_norm"),
        ("indexer_compressor_pool", "indexer_compressor_norm_rope", "indexer_pool_norm_rope"),
        ("compressor_pool", "compressor_norm_rope_cache", "compressor_pool_norm_cache"),
    ):
        left, right = scopes[first], scopes[second]
        assert ast.dump(left.items[0].context_expr.args[0]) == ast.dump(right.items[0].context_expr.args[0])
        merged = copy.deepcopy(left)
        set_keyword(merged, "name_hint", ast.Constant(combined))
        # Norm/cache consumes metadata; the merged producer must retain that edge.
        for item in right.items[0].context_expr.keywords:
            if item.arg == "deps":
                assert not any(item.arg == "deps" for item in merged.items[0].context_expr.keywords)
                merged.items[0].context_expr.keywords.append(copy.deepcopy(item))
        math_body = copy.deepcopy(left.body) + ast.parse("pl.system.bar_all()").body + copy.deepcopy(right.body)
        merged.body = ast.parse("owner_worker = pl.tile.get_block_idx()").body + [
            OwnerIndex().visit(node) for node in math_body
        ]
        replacements[first] = merged
        replacements[second] = None

    for name, task_id in (("indexer_query_proj", "index_query_tid"), ("indexer_query_rope", "index_rope_tid")):
        scope = copy.deepcopy(scopes[name])
        scope.items[0].optional_vars = ast.Name(id=task_id, ctx=ast.Store())
        replacements[name] = scope
    for name, task_id in (
        ("compressor_kv_score", "index_query_tid"),
        ("qproj_b", "index_query_tid"),
        ("qproj_dequant_rms_nope_rope", "index_rope_tid"),
    ):
        scope = copy.deepcopy(scopes[name])
        set_keyword(scope, "deps", ast.List(elts=[ast.Name(id=task_id, ctx=ast.Load())], ctx=ast.Load()))
        replacements[name] = scope

    # The former direct Q_A dependency is now carried transitively by Indexer Q.
    scope = copy.deepcopy(scopes["qproj_a"])
    scope.items[0].optional_vars = None
    replacements["qproj_a"] = scope

    lines = source.splitlines(keepends=True)
    for name in sorted(replacements, key=lambda name: scopes[name].lineno, reverse=True):
        original, replacement = scopes[name], replacements[name]
        text = (
            ""
            if replacement is None
            else textwrap.indent(ast.unparse(ast.fix_missing_locations(replacement)), "    ") + "\n"
        )
        lines[original.lineno - 1 : original.end_lineno] = [text]
    path.write_text("".join(lines))
    count = sum(scope_name(node) is not None for node in ast.walk(ast.parse(path.read_text())))
    assert count == 40, count
    print(path)


if __name__ == "__main__":
    main()
