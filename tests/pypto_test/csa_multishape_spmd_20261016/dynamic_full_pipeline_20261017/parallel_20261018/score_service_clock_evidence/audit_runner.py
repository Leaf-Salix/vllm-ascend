"""标准库静态审阅；不导入runner、torch、PyPTO，不执行设备或lowering。"""

import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path

from prepare_runner import BASE, HELPERS, HERE, ORIGIN, PACKAGE, REPLACEMENTS, REPO


def main():
    runner = HERE / "run_score_diagnostic.py"
    text = runner.read_text()
    restored = text
    assert restored.count(HELPERS) == 1
    restored = restored.replace(HELPERS, "")
    for old, new in reversed(REPLACEMENTS):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    original = (ORIGIN / "run_tick_diagnostic.py").read_text()
    assert restored == original
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    wrapper = (HERE / "run_device.sh").read_text()
    assert wrapper.replace("run_score_diagnostic.py", "run_tick_diagnostic.py") == (ORIGIN / "run_device.sh").read_text()
    tree = ast.parse(text)
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    top_modules = [node.module if isinstance(node, ast.ImportFrom) else node.names[0].name for node in top_imports]
    assert top_modules == ["argparse", "json", "os", "sys", "traceback", "pathlib", "types"]
    score_imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                     and node.module == "score_service_clock"]
    assert len(score_imports) == 1
    assert sorted(alias.name for alias in score_imports[0].names) == [
        "check_probe_padding", "compilation_probe", "export_clock"
    ]
    assert "attention_tick_clock" not in text
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    hook_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "check_probe_padding"]
    assert len(hook_calls) == 1
    assert ast.unparse(hook_calls[0]) == "check_probe_padding(signal, generations[name])"
    assert 'input_sequence=["A", "B", "A"]' in text and 'wait_backoff_ticks=100' in text
    assert 'performance_measured=False' in text and 'eligible_for_performance=False' in text
    assert '(args.history, args.batch) != (131072, 16)' in text
    assert 'not args.baseline_only or args.candidate_source' in text
    assert 'compare_tensor(value, baseline[label][key], 0, 0)' in text
    assert 'signal[0, :, 0] == generations[name]' in text

    snapshot = HERE / "source_snapshot"
    names = sorted(path.name for path in snapshot.glob("*.py"))
    assert len(names) == 33
    assert names == sorted(Path(path).name for path in subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, PACKAGE], cwd=REPO, text=True
    ).splitlines() if path.endswith(".py"))
    sizes = []
    for name in names:
        path = snapshot / name
        expected = subprocess.check_output(["git", "show", f"{BASE}:{PACKAGE}/{name}"], cwd=REPO)
        assert path.read_bytes() == expected
        ast.parse(path.read_text())
        sizes.append({"name": name, "bytes": path.stat().st_size})
    syntax = HERE / "syntax_check"
    syntax.mkdir(exist_ok=True)
    hook = HERE / "score_service_clock.py"
    py_files = [HERE / "prepare_runner.py", HERE / "audit_runner.py", runner]
    if hook.exists():
        py_files.append(hook)
        hook_tree = ast.parse(hook.read_text())
        functions = {node.name: node for node in hook_tree.body if isinstance(node, ast.FunctionDef)}
        for name, arguments in {
            "compilation_probe": ["directory"],
            "export_clock": ["signal", "destination", "phases"],
            "check_probe_padding": ["signal", "generation"],
        }.items():
            assert name in functions
            actual = [arg.arg for arg in functions[name].args.args]
            assert actual == arguments, (name, actual)
    for path in py_files:
        py_compile.compile(str(path), cfile=str(syntax / (path.stem + ".pyc")), doraise=True)
    assert not any(name == "torch" or name.startswith("torch.") or name == "pypto" or name.startswith("pypto.")
                   for name in sys.modules)
    result = {
        "passed": True, "scope": "入口/包装/33源的标准库静态检查；不是完整CPU或设备验证",
        "base_revision": BASE, "source_files": sizes,
        "source_count": 33, "source_direct_bytes_equal_to_base": True,
        "runner_inverse_text_and_ast_equal_to_origin": True,
        "wrapper_only_runner_filename_changed": True,
        "origin_validation_flow_preserved_except_declared_changes": True,
        "active_guard_check": "AIC有CSA_SCORE_PROBE_BEGIN/END，AIV无；两侧CSA_PROFILE_NOTIFY保留",
        "allowed_padding": "所有phase各core列16..19；仅phase34/AIC0..23额外允许列32..63；其余保留位为0，由hook检查",
        "hook_copy_present_and_interface_checked": hook.exists(),
        "syntax_compiled_only": [str(path) for path in py_files],
        "torch_pypto_imported": False, "pypto_compiled": False, "device_submitted": False,
    }
    (HERE / "RUNNER_STATIC_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "source_count": 33, "hook_present": hook.exists(),
                      "pypto_compiled": False, "device_submitted": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
