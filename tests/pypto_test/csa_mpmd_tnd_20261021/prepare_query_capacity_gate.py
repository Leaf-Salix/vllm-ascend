# SPDX-License-Identifier: Apache-2.0
"""在同一TND多task入口按Tensor容量选择独立query阶段或局部MIX。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    source_path = target / "entry.py"
    source = source_path.read_text()
    root = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_tnd"
    )
    scope = next(node for node in root.body if scope_name(node) == "indexer_query_chain")
    reference = ast.parse((here / "experiments/v17_upstream_fixes/operator/entry.py").read_text())
    old_scopes = {scope_name(node): node for node in ast.walk(reference) if scope_name(node)}
    names = (
        "indexer_query_proj",
        "indexer_query_rope",
        "indexer_query_hadamard",
        "indexer_query_quant",
        "indexer_weights_proj",
        "indexer_weights_reduce",
        "indexer_head_coefficients",
    )
    small = ast.parse(
        textwrap.dedent("""
        query_rope_workers = VECTORS
        query_hadamard_workers = GROUPS
        weight_workers = GROUPS
        if LIMIT_EMPTY_WORKERS:
            query_rope_workers = pl.min(
                VECTORS, ((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
            )
            query_hadamard_workers = pl.min(GROUPS, tokens)
            weight_workers = pl.min(GROUPS, WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE))
    """)
    ).body + [copy.deepcopy(old_scopes[name]) for name in names]
    gate = ast.If(test=ast.Name(id="GROUP_QUERY", ctx=ast.Load()), body=[copy.deepcopy(scope)], orelse=small)
    lines = source.splitlines(keepends=True)
    lines[scope.lineno - 1 : scope.end_lineno] = [
        textwrap.indent(ast.unparse(ast.fix_missing_locations(gate)), "    ") + "\n"
    ]
    source = "".join(lines)
    needle = "    LIMIT_EMPTY_WORKERS: pl.constexpr = False,\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "    GROUP_QUERY: pl.constexpr = True,\n")
    source = source.replace(
        "from .indexer_group_stages import indexer_group_plan_stage",
        "from .indexer_group_stages import indexer_group_plan_stage, indexer_head_coefficients_stage",
    )
    imports = """from .indexer_query_stages import (
    DEQUANT_T_TILE, DQ_ROPE_H_TILE, IDX_N_HEADS, MM_ROW_TILE, WEIGHTS_OK,
    indexer_query_project_stage, indexer_query_dequant_rope_stage,
    indexer_query_hadamard_stage, indexer_query_quant_stage,
    indexer_weights_project_stage, indexer_weights_reduce_stage,
)
"""
    source = source.replace("from .indexer_publish_stages", imports + "from .indexer_publish_stages", 1)
    source_path.write_text(source)

    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    policy = policy.replace(
        "NONEMPTY_WORKER_MAX_TOKENS = 64\n",
        "NONEMPTY_WORKER_MAX_TOKENS = 64\n"
        "# 两个完整M32容量起才采用局部query MIX；小容量已测整体融合回退。\n"
        "GROUP_QUERY_MIN_TOKENS = 64\n",
    )
    policy = policy.replace(
        "    limit_empty_workers: bool = False\n",
        "    limit_empty_workers: bool = False\n    group_query: bool = True\n",
    )
    policy = policy.replace(
        '            "LIMIT_EMPTY_WORKERS": self.limit_empty_workers,\n',
        '            "LIMIT_EMPTY_WORKERS": self.limit_empty_workers,\n            "GROUP_QUERY": self.group_query,\n',
    )
    policy = policy.replace(
        "    return CSAMPMDPolicy(\n",
        "    return CSAMPMDPolicy(\n        group_query=tokens >= GROUP_QUERY_MIN_TOKENS,\n",
    )
    assert policy.count('"GROUP_QUERY"') == 1 and policy.count("group_query=tokens") == 1
    policy_path.write_text(policy)
    binding_path = target / "binding.py"
    binding = binding_path.read_text()
    needle = '"query_signal": ((QUERY_SIGNAL_COUNT,), torch.int32),'
    assert binding.count(needle) == 1
    binding = binding.replace(
        needle, '"query_signal": ((QUERY_SIGNAL_COUNT if self.policy.group_query else 1,), torch.int32),'
    )
    binding_path.write_text(binding)
    print(target)


if __name__ == "__main__":
    main()
