"""仅补冻结hook的接口/语法检查；不重复runner审计、不导入hook或设备库。"""

import ast
import json
import py_compile
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/"
              "tests/pypto_test/score_service_clock_20261018/score_service_clock.py")


def main():
    destination = HERE / SOURCE.name
    data = SOURCE.read_bytes()
    if destination.exists():
        assert destination.read_bytes() == data, "已存在的冻结hook不能覆盖"
    else:
        destination.write_bytes(data)
    assert destination.read_bytes() == SOURCE.read_bytes()
    tree = ast.parse(data.decode())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    contracts = {"compilation_probe": ["directory"],
                 "export_clock": ["signal", "destination", "phases"],
                 "check_probe_padding": ["signal", "generation"]}
    for name, expected in contracts.items():
        function = functions[name]
        assert [argument.arg for argument in function.args.args] == expected
        assert not function.args.vararg and not function.args.kwarg
    syntax = HERE / "syntax_check"
    syntax.mkdir(exist_ok=True)
    py_compile.compile(str(destination), cfile=str(syntax / "score_service_clock.pyc"), doraise=True)
    assert not any(name == "torch" or name.startswith("torch.") or name == "pypto" or name.startswith("pypto.")
                   for name in sys.modules)
    report = {"passed": True, "source": str(SOURCE), "copy": str(destination), "bytes": len(data),
              "direct_bytes_equal": True, "interfaces": contracts, "standard_library_py_compile": True,
              "hook_imported": False, "torch_pypto_imported": False, "pypto_compiled": False,
              "device_submitted": False, "prior_runner_checks_or_models_rerun": False}
    (HERE / "HOOK_INTERFACE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    audit_path = HERE / "RUNNER_STATIC_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    assert audit["passed"]
    audit["hook_copy_present_and_interface_checked"] = True
    audit["hook_interface_audit"] = str(HERE / "HOOK_INTERFACE_AUDIT.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
