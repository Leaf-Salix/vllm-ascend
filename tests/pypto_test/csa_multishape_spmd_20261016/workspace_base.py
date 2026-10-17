# SPDX-License-Identifier: Apache-2.0
"""归因实验：以真实 workspace 基址重建临时 GM 指针，减少跨阶段活跃指针。

不改变数学、调度、数据布局或运行时参数数目；只在当前编译进程改写指针表达式。
用实际准备对象逐项核对地址后才允许启用。未作为生产 ABI 实现。
"""

import ast
import contextlib
import json
import re
from pathlib import Path

DTYPE_BYTES = {"INT8": 1, "BF16": 2, "FP16": 2, "FP32": 4, "INT32": 4, "INT64": 8}


def aliases(case):
    case = Path(case)
    layout = json.loads((case / "layout.json").read_text())
    contract = json.loads((case / "entry.bindings.json").read_text())["_entry_contract"]
    refs = {row["name"]: row["reference"] for row in
            contract["ordered_inputs"] + contract["derived_native_views"]}
    offset = layout["workspace"]["sections"]["progress"]["offset_bytes"]
    result = {}
    for name, ref in refs.items():
        allocation = ref.get("allocation")
        if allocation is None:
            byte_offset = ref["offset_bytes"]
        elif allocation.startswith("argument:"):
            continue
        else:
            byte_offset = (layout["workspace"]["temporaries"][allocation]["offset_bytes"]
                           + ref["offset"] * DTYPE_BYTES[ref["dtype"]])
        if name != "progress":
            result[name] = byte_offset - offset
    return result


def check_addresses(case, prepared):
    offsets = aliases(case)
    tensors = dict(zip(prepared.param_names, prepared.core_args))
    base = tensors["progress"].data_ptr()
    for name, offset in offsets.items():
        if tensors[name].data_ptr() != base + offset:
            raise ValueError("实际临时 GM 指针不符合 arena 别名合同：" + name)
    return {"aliases_checked": len(offsets), "actual_addresses_checked": True}


def rebase(source, case):
    module = ast.parse((Path(case) / "entry_jit.py").read_text())
    offsets = aliases(case)
    signatures = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    audit = []
    for match in reversed(signatures):
        role = match.group(1)
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "csa_static_" + role)
        declarations = match.group(2).split(",")
        identifiers = [item.strip().split()[-1].lstrip("*") for item in declarations]
        positions = {arg.arg: index for index, arg in enumerate(fn.args.args)}
        base = identifiers[positions["progress"]]
        if "__gm__ int32_t*" not in declarations[positions["progress"]]:
            raise ValueError("进度基址 C++ ABI 类型改变")
        definitions = []
        for name, offset in offsets.items():
            index = positions[name]
            decl = declarations[index]
            if "__gm__" not in decl:
                raise ValueError("临时 GM 参数类型改变")
            var = identifiers[index]
            typ = decl.strip().removesuffix(var).strip()
            declarations[index] = decl.removesuffix(var) + "unused_" + var
            definitions.append(f"  {typ} {var} = ({typ})((__gm__ unsigned char*){base} + ({offset}));")
        header = f"AICORE void csa_static_{role}({','.join(declarations)}) {{"
        source = source[:match.start()] + header + "\n" + "\n".join(definitions) + source[match.end():]
        audit.append({"role": role, "rebased_pointers": len(definitions),
                      "math_and_schedule_unchanged": True})
    return source, audit


@contextlib.contextmanager
def workspace_base_codegen(case, prepared, output):
    from pypto.backend import pto_backend

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = check_addresses(case, prepared)
    original = pto_backend._compile_pto_module

    def compile_rebased(pto, name, output_dir, *args, **kwargs):
        source = original(pto, name, output_dir, *args, **kwargs)
        if name != "csa_static_group":
            return source
        updated, audit = rebase(source, case)
        (output / "original.cpp").write_text(source)
        (output / "rebased.cpp").write_text(updated)
        (output / "audit.json").write_text(json.dumps(
            {"address_validation": checks, "functions": audit}, ensure_ascii=False, indent=2) + "\n")
        return updated

    pto_backend._compile_pto_module = compile_rebased
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original
