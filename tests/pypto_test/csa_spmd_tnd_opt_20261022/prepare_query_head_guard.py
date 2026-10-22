# SPDX-License-Identifier: Apache-2.0
"""Give the measured Query head orientation an independent constexpr domain."""

import ast
import hashlib
import json
import runpy
import shutil
from pathlib import Path

FLAG = "QUERY_CONTIGUOUS_HEADS"


def edit_functions(source, names, edit):
    selected = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for fn in reversed(selected):
        original = ast.get_source_segment(source, fn)
        edit(fn)
        fn.decorator_list = []  # Original decorators are outside the replaced segment.
        source = source.replace(original, ast.unparse(ast.fix_missing_locations(fn)), 1)
    ast.parse(source)
    return source


def add_flag(fn):
    assert FLAG not in [arg.arg for arg in fn.args.args]
    fn.args.args.append(ast.arg(arg=FLAG, annotation=ast.parse("pl.constexpr", mode="eval").body))
    fn.args.defaults.append(ast.Constant(value=False))


def forward(fn, target):
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == target]
    assert len(calls) == 1, (fn.name, target, len(calls))
    calls[0].args.append(ast.Name(id=FLAG, ctx=ast.Load()))


def transform_query(source):
    def edit(fn):
        old_args = ", ".join(n.arg for n in fn.args.args)
        add_flag(fn)
        assert len(fn.body) == 2 and isinstance(fn.body[0], ast.If)
        fallback = ast.parse(f"q_flat = _q_dequant_rope_original({old_args})").body
        fn.body[0] = ast.If(test=ast.Name(id=FLAG, ctx=ast.Load()), body=[fn.body[0]], orelse=fallback)

    return edit_functions(source, ["q_dequant_rope_stage"], edit)


def transform_vector(source):
    def edit(fn):
        add_flag(fn)
        forward(fn, "q_dequant_rope_stage")

    return edit_functions(source, ["vector_schedule"], edit)


def transform_entry(source):
    def edit(fn):
        add_flag(fn)
        forward(fn, "vector_schedule" if fn.name == "_csa_core_group" else "_csa_core_group")

    return edit_functions(source, ["_csa_core_group", "decode_csa_spmd"], edit)


def transform_policy(source):
    field = "    prefix_branch_groups: int = 4\n"
    constant = '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n'
    changes = {
        field: field + "    query_contiguous_heads: bool = False\n",
        constant: constant + '            "QUERY_CONTIGUOUS_HEADS": self.query_contiguous_heads,\n',
        "    return CSASPMDPolicy(\n": "    return CSASPMDPolicy(\n"
        "        # Independent experimental choice in the existing Q_B/O_A resource interval.\n"
        "        # Dynamic token tails still execute the entire original Query helper.\n"
        "        query_contiguous_heads=(\n"
        "            long_history and group == 6\n"
        "            and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS\n"
        "        ),\n",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/query_contiguous_heads_v4"
    expected = json.loads((here / "query_contiguous_heads_v4.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/query_head_guard"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    transforms = {
        "query_projection_stages.py": transform_query,
        "vector_schedule.py": transform_vector,
        "entry.py": transform_entry,
        "policy.py": transform_policy,
    }
    for name, change in transforms.items():
        (target / name).write_text(change((parent / name).read_text()))
    policy = runpy.run_path(str(target / "policy.py"))["select_policy"]
    descriptors = []
    for label, history, batch in (("b16", 131072, 16), ("short_b32", 8192, 32)):
        base = (
            here.parent / "csa_spmd_tnd_20261020/compile_descriptors" / f"shared_combined_v122_h{history}_b{batch}.json"
        )
        descriptor = json.loads(base.read_text())
        actual = policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6).constexprs()
        assert {k: v for k, v in actual.items() if k != FLAG} == descriptor["policy"]
        descriptor["policy"] = actual
        path = here / "descriptors" / f"query_head_guard_{label}.json"
        with path.open("x") as stream:
            stream.write(json.dumps(descriptor, indent=2) + "\n")
        descriptors.append(str(path))
    report = dict(
        source=str(target),
        parent="query_contiguous_heads_v4",
        reference="frozen_tnd_v122",
        modified_files=list(transforms),
        descriptors=descriptors,
        domain="long history G6 and 64<=token capacity<=96; original helper on device token tails",
        flag=FLAG,
        adopted=False,
        device_validated=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "query_head_guard.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
