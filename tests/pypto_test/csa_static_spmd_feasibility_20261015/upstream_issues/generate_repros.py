"""从现有组件回归提取上游 issue 的独立 CPU 复现。"""

import ast
import json
import sys
import textwrap
from pathlib import Path


def extract(path, name):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = source.splitlines()
    return textwrap.dedent("\n".join(lines[first - 1:node.end_lineno])).replace("(self, ", "(")


def main():
    root = Path(sys.argv[1])
    output = Path(__file__).resolve().parent / "repro"
    output.mkdir(exist_ok=True)
    group = root / "tests/ut/ir/transforms/test_expand_mixed_kernel_a2a3.py"
    codegen_path = root / "tests/ut/codegen/test_memory_planner_switch.py"
    license_header = "\n".join(group.read_text().splitlines()[:8])
    header = license_header + """

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()
"""
    specs = [
        ("p1_group_inout", group, ["_run_pipeline"], [
            "test_hand_written_group_rebinds_shared_inout_results",
            "test_hand_written_group_submit_rebinds_tuple_writeback_no_dep"]),
        ("p4_ptoas_branch_view", codegen_path, ["_run_pipeline", "_codegen"], [
            "test_ptoas_reshape_reads_phi_handle_after_branch"]),
        ("p5_ptoas_gather_phi", codegen_path, ["_run_pipeline", "_codegen"], [
            "test_ptoas_conditional_gather_preserves_one_mat_buffer"]),
        ("p7_ptoas_column_carry", codegen_path, ["_run_pipeline", "_codegen"], [
            "test_ptoas_column_carry_move_uses_destination_view"]),
    ]
    manifest = []
    for key, source, helpers, tests in specs:
        chunks = [header]
        for name in helpers + tests:
            chunks.append(extract(source, name).replace("program: ir.Program = ElementwiseAdd", "program: ir.Program"))
        text = "\n\n".join(chunks) + "\n"
        ast.parse(text)
        target = output / ("test_" + key + ".py")
        target.write_text(text)
        manifest.append({"id": key, "source": str(source.relative_to(root)),
                         "tests": tests, "repro": str(target.relative_to(output.parent))})
    (output.parent / "repro_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(specs), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
