# SPDX-License-Identifier: Apache-2.0
"""只读核对七档原型参数来源和两类核的实际引用；不编译或启动设备。"""

import ast
import json
from collections import Counter
from pathlib import Path


def audit_case(directory):
    contract = json.loads((directory / "entry.bindings.json").read_text())["_entry_contract"]
    inputs = contract["ordered_inputs"]
    names = {item["name"] for item in inputs}
    roots = Counter()
    temporaries = Counter()
    protocol = []
    for item in inputs:
        allocation = item["reference"].get("allocation")
        if allocation is None:
            protocol.append(item["name"])
        elif allocation.startswith("argument:"):
            roots[allocation] += 1
        else:
            temporaries[allocation] += 1
    body_usage = {}
    for node in ast.walk(ast.parse((directory / "entry.py").read_text())):
        if not isinstance(node, ast.FunctionDef) or node.name not in ("csa_static_aic", "csa_static_aiv"):
            continue
        # return 仅转交别名，不据此把两半共享签名中的全部参数算成计算引用。
        loads = {child.id for statement in node.body if not isinstance(statement, ast.Return)
                 for child in ast.walk(statement)
                 if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        body_usage[node.name] = sorted(names & loads)
    union = set().union(*(set(items) for items in body_usage.values()))
    assert len(inputs) == contract["leaf_tensor_count"] == len(names)
    return {
        "case": directory.name,
        "tensor_parameters": len(inputs),
        "native_descriptors": sum(roots.values()),
        "distinct_native_roots": len(roots),
        "temporary_descriptors": sum(temporaries.values()),
        "distinct_temporary_allocations": len(temporaries),
        "protocol_descriptors": protocol,
        "repeated_native_views": {k: v for k, v in roots.items() if v > 1},
        "repeated_temporary_views": {k: v for k, v in temporaries.items() if v > 1},
        "body_parameter_counts": {k: len(v) for k, v in body_usage.items()},
        "not_loaded_by_either_body": sorted(names - union),
        "body_parameters": body_usage,
        "scope": "静态源码名称引用及绑定来源审计，不证明 Native 实际指针均不同，也不证明核内 GM 视图可重建",
    }


def main():
    root = Path(__file__).resolve().parent
    cases = [audit_case(path.parent) for path in sorted((root / "static").glob("h*/entry.bindings.json"))]
    result = {"device_execution": False, "dependency_changes": False, "cases": cases}
    (root / "static/core_parameter_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in case.items() if k != "body_parameters"} for case in cases],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
