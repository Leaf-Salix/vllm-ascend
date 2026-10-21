# SPDX-License-Identifier: Apache-2.0
"""HC局部MIX：按真实行owner等待，保留大容量8行gate/norm及其余独立task。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_output_group import make_function, root_function
from prepare_priority_fusion import scope_name


def parse(source):
    return ast.parse(source).body


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    shutil.copyfile(here / "hc_row_stages_source.py", target / "hc_rows.py")
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    root = root_function(source, "decode_csa_tnd")
    arguments = {a.arg: a for a in root.args.args}
    new_args = parse("""
def f(hc_signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
      HC_GROUPS: pl.constexpr, HC_ROW_TILE: pl.constexpr): pass
""")[0].args.args
    arguments.update({a.arg: a for a in new_args})
    cube_params = ["x_fp32", "hc_fn", "partials", "hc_signal", "HC_GROUPS", "HC_ROW_TILE"]
    cube_outputs = ["partials", "hc_signal"]
    cube_body = parse("""
worker = pl.tile.get_block_idx()
tokens = pl.tensor.dim(x_fp32, 0)
(hc_signal, generation) = begin_invocation(hc_signal, worker)
units = (tokens + 15) // 16 * 4
for unit in pl.range(worker, units, HC_GROUPS):
    first = unit // 4 * 16
    hc_signal = wait_row_owners(hc_signal, H_WIDE, 3 * HC_GROUPS, generation,
        first, pl.min(16, tokens - first), HC_ROW_TILE, 2 * HC_GROUPS, HC_GROUPS, 1)
partials = hc_project_range(x_fp32, hc_fn, partials, worker, HC_GROUPS)
hc_signal = publish_phase(hc_signal, H_PROJECT, worker, 3 * HC_GROUPS, generation)
""")
    vector_params = [
        "x",
        "hc_scale",
        "hc_base",
        "norm_w",
        "partials",
        "x_fp32",
        "inv",
        "widen_tail",
        "raw",
        "pre",
        "post",
        "post_tail",
        "mixed",
        "normed",
        "norm_tail",
        "hc_signal",
        "HC_GROUPS",
        "HC_ROW_TILE",
        "HC_CONSUMER_REDUCE",
    ]
    vector_outputs = [
        "x_fp32",
        "inv",
        "widen_tail",
        "raw",
        "pre",
        "post",
        "post_tail",
        "mixed",
        "normed",
        "norm_tail",
        "hc_signal",
    ]
    vector_body = parse("""
pl.func_attr({'split_aiv': True})
worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
core = HC_GROUPS + worker
tokens = pl.tensor.dim(x, 0)
(hc_signal, generation) = begin_invocation(hc_signal, core)
(x_fp32, inv) = row_hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, 2 * HC_GROUPS, HC_ROW_TILE)
hc_signal = publish_phase(hc_signal, H_WIDE, core, 3 * HC_GROUPS, generation)
for block in pl.range(worker, (tokens + 15) // 16, 2 * HC_GROUPS):
    hc_signal = wait_row_owners(hc_signal, H_PROJECT, 3 * HC_GROUPS, generation,
        block * 4, 4, 1, HC_GROUPS, 0, 1)
raw = hc_reduce_range(partials, raw, worker, 2 * HC_GROUPS)
hc_signal = publish_phase(hc_signal, H_RAW, core, 3 * HC_GROUPS, generation)
if tokens < 8:
    for block in pl.range(worker, tokens, 2 * HC_GROUPS):
        hc_signal = wait_row_owners(hc_signal, H_RAW, 3 * HC_GROUPS, generation,
            block, 1, 16, 2 * HC_GROUPS, HC_GROUPS, 1)
    (pre, post) = row_hc_gates_range(inv, raw, hc_scale, hc_base, pre, post, post_tail,
        worker, 2 * HC_GROUPS, partials, HC_CONSUMER_REDUCE, 1)
    pl.system.bar_all()
    (mixed, normed) = row_hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail,
        worker, 2 * HC_GROUPS, 1)
else:
    for block in pl.range(worker, (tokens + 7) // 8, 2 * HC_GROUPS):
        first = block * 8
        hc_signal = wait_row_owners(hc_signal, H_RAW, 3 * HC_GROUPS, generation,
            first, pl.min(8, tokens - first), 16, 2 * HC_GROUPS, HC_GROUPS, 1)
    (pre, post) = hc_gates_range(inv, raw, hc_scale, hc_base, pre, post, post_tail,
        worker, 2 * HC_GROUPS, partials, HC_CONSUMER_REDUCE)
    pl.system.bar_all()
    (mixed, normed) = hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail,
        worker, 2 * HC_GROUPS)
""")
    # tail scratch保留InOut合同，只有唯一尾块owner会写它。
    # 全部InOut按声明顺序返回，匹配当前PyPTO metadata传播规则。
    common_params = list(dict.fromkeys(cube_params + vector_params))
    common_outputs = [p for p in common_params if p in cube_outputs or p in vector_outputs]
    chain_body = parse(
        "with pl.cluster(name_hint='hc_prefix_chain'):\n"
        f"    ({', '.join(cube_outputs)}) = hc_prefix_cube({', '.join(cube_params)})\n"
        f"    ({', '.join(vector_outputs)}) = hc_prefix_vector({', '.join(vector_params)})\n"
    )
    imports = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    imports += parse("""
from .hc_rows import row_hc_widen_rms_range, row_hc_gates_range, row_hc_mix_norm_range
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
HC_SIGNAL_WORDS = pl.dynamic('CSA_HC_PREFIX_SIGNAL_WORDS')
H_WIDE = 1
H_PROJECT = 2
H_RAW = 3
""")
    functions = [
        make_function("hc_prefix_cube", cube_params, arguments, cube_outputs, cube_body, "pl.jit.incore"),
        make_function("hc_prefix_vector", vector_params, arguments, vector_outputs, vector_body, "pl.jit.incore"),
        make_function(
            "hc_prefix_chain", common_params, arguments, common_outputs, chain_body, "pl.jit.inline(auto_scope=False)"
        ),
    ]
    (target / "hc_pipeline.py").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n"""独立HC局部流水；信号一次清零、按调用代际推进。"""\n'
        + ast.unparse(ast.fix_missing_locations(ast.Module(body=imports + functions, type_ignores=[])))
        + "\n"
    )
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    replacement = parse(
        "with pl.spmd(HC_GROUPS, name_hint='hc_prefix_chain', allow_early_resolve=True, sync_start=True):\n"
        f"    ({', '.join(common_outputs)}) = hc_prefix_chain({', '.join(common_params)})\n"
    )[0]
    gate.body = [
        replacement if scope_name(n) == "hc_widen_rms" else n
        for n in gate.body
        if scope_name(n) not in ("hc_linear", "hc_reduce", "hc_gate_mix_norm")
    ]
    required = len(root.args.args) - len(root.args.defaults)
    root.args.args.insert(required, arguments["hc_signal"])
    root.args.args += [arguments["HC_GROUPS"], arguments["HC_ROW_TILE"]]
    root.args.defaults += [ast.Constant(24), ast.Constant(8)]
    root.body[-1].value.elts.append(ast.Name(id="hc_signal", ctx=ast.Load()))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .hc_pipeline import HC_SIGNAL_WORDS, hc_prefix_chain\n",
        1,
    )
    entry_path.write_text(source)
    policy_path = target / "policy.py"
    source = policy_path.read_text()
    source = source.replace(
        "    fused_dag: bool = True\n",
        "    fused_dag: bool = True\n    hc_groups: int = 24\n    hc_row_tile: int = 8\n",
        1,
    )
    source = source.replace(
        '            "FUSED_DAG": self.fused_dag,\n',
        '            "FUSED_DAG": self.fused_dag,\n'
        '            "HC_GROUPS": self.hc_groups,\n'
        '            "HC_ROW_TILE": self.hc_row_tile,\n',
        1,
    )
    source = source.replace(
        "    return CSAMPMDPolicy(\n",
        "    hc_groups = min(24, max(4, 4 * ((tokens + 15) // 16)))\n"
        "    hc_row_tile = next((tile for tile in (1, 2, 4, 8) if tokens <= 2 * hc_groups * tile), 8)\n"
        "    return CSAMPMDPolicy(\n        hc_groups=hc_groups,\n        hc_row_tile=hc_row_tile,\n",
        1,
    )
    policy_path.write_text(source)
    binding_path = target / "binding.py"
    source = binding_path.read_text()
    needle = '            "query_signal": ((1,), torch.int32),\n'
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        needle
        + '            "hc_signal": ((4 * 3 * self.policy.hc_groups * 128'
        + " if self.policy.fused_dag else 1,), torch.int32),\n",
        1,
    )
    source = source.replace('if name in ("middle_signal",', 'if name in ("hc_signal", "middle_signal",', 1)
    binding_path.write_text(source)
    for path in (entry_path, policy_path, binding_path, target / "hc_pipeline.py"):
        ast.parse(path.read_text())
    print(target)


if __name__ == "__main__":
    main()
