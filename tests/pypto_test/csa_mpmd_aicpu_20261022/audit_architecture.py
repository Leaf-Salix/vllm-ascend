# SPDX-License-Identifier: Apache-2.0
"""检查算法包禁止的阶段软同步及实际编排；不扫描二进制hash、不访问设备。"""

import argparse
import ast
import json
import re
from pathlib import Path


def audit_source(package):
    failures = []
    forbidden = {
        "signal_wait",
        "signal_notify",
        "csa_signal_wait",
        "csa_signal_notify",
        "wait_row_owners",
        "publish_phase",
        "begin_invocation",
    }
    task_names = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if name in forbidden:
                failures.append(f"{path.name}:{node.lineno}: {name}")
            if name == "spmd":
                task_names.extend(
                    k.value.value for k in node.keywords if k.arg == "name_hint" and isinstance(k.value, ast.Constant)
                )
    required = {
        "hc_widen_rms",
        "hc_linear",
        "qproj_a",
        "qproj_b",
        "indexer_score",
        "indexer_topk",
        "sparse_attention",
        "o_proj_a",
        "o_quant",
        "o_proj_b",
        "hc_post",
    }
    failures.extend(f"missing independent stage: {name}" for name in sorted(required - set(task_names)))
    failures.extend(
        f"remaining resident task: {name}"
        for name in task_names
        if name.endswith("_chain") or name == "csa_row_offsets"
    )
    return {
        "scope": "source calls and stage declarations; codegen and device checks remain separate",
        "task_names": task_names,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--generated", type=Path)
    args = parser.parse_args()
    result = audit_source(args.package)
    if args.generated:
        paths = list(args.generated.glob("decode_csa_tnd_*/orchestration/decode_csa_tnd.cpp"))
        assert len(paths) == 1, paths
        code = paths[0].read_text()
        result["generated_orchestration"] = str(paths[0])
        result["submit_sites"] = len(re.findall(r"rt_submit_\w+\(", code))
        result["aicpu_tensor_reads"] = code.count("get_tensor_data<")
        result["aicpu_tensor_writes"] = code.count("set_tensor_data<")
        assert result["aicpu_tensor_reads"] and result["aicpu_tensor_writes"]
        assert not re.search(r"// (?:Spmd|Group) (?:csa_row_offsets|.*_chain)", code)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
