# SPDX-License-Identifier: Apache-2.0
"""Limit O_A quantization reuse to the supported single-request small tile.

The user explicitly permits a separate B1 policy. The query capacity is at most
six in the current decode contract, so this domain fits one eight-row tile.
"""

import ast
import hashlib
import json
import runpy
import shutil
from pathlib import Path

from prepare_query_head_guard import edit_functions

FLAG = "OUTPUT_QUANT_REUSE"


def add_flag(fn):
    assert FLAG not in [a.arg for a in fn.args.args]
    fn.args.args.append(ast.arg(arg=FLAG, annotation=ast.parse("pl.constexpr", mode="eval").body))
    fn.args.defaults.append(ast.Constant(value=False))


def forward(fn, targets):
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in targets]
    assert len(calls) == len(targets)
    for call in calls:
        call.args.append(ast.Name(id=FLAG, ctx=ast.Load()))


def transform_output(source):
    def edit(fn):
        add_flag(fn)
        if fn.name == "o_projection_group_quant_stage":
            forward(fn, ["o_projection_groups_quant_stage"])
            return
        for block in ast.walk(fn):
            body = getattr(block, "body", None)
            if not isinstance(body, list):
                continue
            for index, node in enumerate(body):
                if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "oc_q":
                    assert ast.unparse(node.value) == "oc_amax"
                    fallback = ast.parse("""
oc_q = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
""").body
                    body[index] = ast.If(test=ast.Name(id=FLAG, ctx=ast.Load()), body=[node], orelse=fallback)
                    return
        raise AssertionError("Missing quant input")

    return edit_functions(source, ["o_projection_groups_quant_stage", "o_projection_group_quant_stage"], edit)


def transform_vector(source):
    def edit(fn):
        add_flag(fn)
        forward(fn, ["o_projection_groups_quant_stage", "o_projection_group_quant_stage"])

    return edit_functions(source, ["vector_schedule"], edit)


def transform_entry(source):
    def edit(fn):
        add_flag(fn)
        forward(fn, ["vector_schedule" if fn.name == "_csa_core_group" else "_csa_core_group"])

    return edit_functions(source, ["_csa_core_group", "decode_csa_spmd"], edit)


def transform_policy(source):
    field = "    prefix_branch_groups: int = 4\n"
    constant = '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n'
    changes = {
        field: field + "    output_quant_reuse: bool = False\n",
        constant: constant + '            "OUTPUT_QUANT_REUSE": self.output_quant_reuse,\n',
        "    return CSASPMDPolicy(\n": "    return CSASPMDPolicy(\n"
        "        # Single-request decode fits one eight-row output quantization tile.\n"
        "        output_quant_reuse=(requests == 1 and tokens <= 8),\n",
    }
    for old, new in changes.items():
        assert source.count(old) == 1
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/output_quant_reuse"
    expected = json.loads((here / "output_quant_reuse.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/output_quant_guard"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    transforms = {
        "output_projection_a_stages.py": transform_output,
        "vector_schedule.py": transform_vector,
        "entry.py": transform_entry,
        "policy.py": transform_policy,
    }
    for name, change in transforms.items():
        (target / name).write_text(change((parent / name).read_text()))
    policy = runpy.run_path(str(target / "policy.py"))["select_policy"]
    descriptors = []
    for batch in (1, 16):
        base = here.parent / "csa_spmd_tnd_20261020/compile_descriptors" / f"shared_combined_v122_h131072_b{batch}.json"
        descriptor = json.loads(base.read_text())
        actual = policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=131078).constexprs()
        assert {k: v for k, v in actual.items() if k != FLAG} == descriptor["policy"]
        descriptor["policy"] = actual
        path = here / "descriptors" / f"output_quant_guard_b{batch}.json"
        with path.open("x") as stream:
            stream.write(json.dumps(descriptor, indent=2) + "\n")
        descriptors.append(str(path))
    report = dict(
        source=str(target),
        parent="output_quant_reuse",
        reference_kind="frozen_tnd_v122",
        modified_files=list(transforms),
        descriptors=descriptors,
        domain="requests == 1 and token capacity <= 8; existing decode contract further limits query to six",
        flag=FLAG,
        adopted=False,
        device_validated=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        limitations="Requires full enabled/disabled generated program comparison; not a nine-case acceptance",
    )
    with (here / "output_quant_guard.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
