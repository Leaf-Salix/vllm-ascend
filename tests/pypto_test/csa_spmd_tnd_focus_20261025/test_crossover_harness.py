# SPDX-License-Identifier: Apache-2.0
"""Execute isolated harness AST with CPU fakes; never import a device runtime."""

import ast
import copy
import json
import subprocess
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from prepare_crossover_harness import transform

HERE = Path(__file__).resolve().parent
OLD = (HERE / "frozen_v2/run_pto.py").read_text()
NEW = (HERE / "frozen_v3/run_pto.py").read_text()
OLD_TREE, NEW_TREE = ast.parse(OLD), ast.parse(NEW)


def find(tree, predicate):
    return next(node for node in ast.walk(tree) if predicate(node))


def function(tree, name):
    return find(tree, lambda node: isinstance(node, ast.FunctionDef) and node.name == name)


def sample_loop(tree, count):
    return find(
        tree,
        lambda node: isinstance(node, ast.For)
        and ast.unparse(node.target) == "sample"
        and ast.unparse(node.iter) == f"range({count})",
    )


def execute_node(node, namespace):
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual frozen_v3 AST>", "exec"), namespace)


def source_block(source, begin, end):
    start = source.index(begin)
    return textwrap.dedent(source[start : source.index(end, start)])


def order_sides(order):
    namespace = {}
    execute_node(function(NEW_TREE, "preparation_sides"), namespace)
    return namespace["preparation_sides"](order)


class CrossoverHarnessTest(unittest.TestCase):
    def test_generator_and_validator_snapshot(self):
        self.assertEqual(transform(OLD), NEW)
        self.assertEqual((HERE / "complete_l1_v2.py").read_bytes(), (HERE / "frozen_v3/complete_l1_v2.py").read_bytes())
        self.assertEqual(
            (HERE / "frozen_v2/complete_l1.py").read_bytes(), (HERE / "frozen_v3/complete_l1.py").read_bytes()
        )
        self.assertIn("frozen_v3/", (HERE / ".gitignore").read_text().splitlines())

    def test_fixture_snapshot_compare_and_restore_sites_unchanged(self):
        for name in ("snapshot", "compare", "check_case", "select_case"):
            self.assertEqual(ast.dump(function(OLD_TREE, name)), ast.dump(function(NEW_TREE, name)), name)
        for source in (OLD, NEW):
            ast.parse(source_block(source, '            layer, report["weights"]', "            entries, guards ="))
        self.assertEqual(
            source_block(OLD, '            layer, report["weights"]', "            entries, guards ="),
            source_block(NEW, '            layer, report["weights"]', "            entries, guards ="),
        )

        def restore_calls(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "restore"]

        self.assertEqual(restore_calls(OLD_TREE), restore_calls(NEW_TREE))

    def test_sampling_warmup_and_post_checks_ast_unchanged(self):
        for count in (3, 10):
            self.assertEqual(ast.dump(sample_loop(OLD_TREE, count)), ast.dump(sample_loop(NEW_TREE, count)))
        predicate = lambda n: isinstance(n, ast.For) and ast.unparse(n.target) == "(side, graph)"
        self.assertEqual(ast.dump(find(OLD_TREE, predicate)), ast.dump(find(NEW_TREE, predicate)))
        self.assertEqual(NEW.count("pypto.torch.begin_dfx()"), 1)
        self.assertEqual(NEW.count("pypto.torch.end_dfx()"), 1)

    def test_actual_package_mount_and_binding_order(self):
        mount = source_block(NEW, "            package_specs =", "        control_name =")
        binding = source_block(NEW, "            entries, guards =", '            report["policies"]')
        for order in ("candidate-first", "reference-first"):
            sides = order_sides(order)
            trace = {"package_mount": [], "binding": []}
            observed = []
            namespace = {
                "args": types.SimpleNamespace(candidate=Path("candidate"), baseline_only=False, family="tnd"),
                "packages": Path("frozen_v3/pypto"),
                "reference_package": "deepseek_v4_flash_csa_spmd_tnd",
                "prefix": "namespace.",
                "prepare_sides": sides,
                "report": {"preparation_trace": trace},
                "mount_package": lambda path, name: name,
                "control_name": None,
                "native": object(),
                "max_query_tokens": 6,
                "case": {"seq_lens": [131078]},
            }
            exec(mount, namespace)

            def import_module(name, observed=observed):
                def construct(*args, **kwargs):
                    observed.append("candidate" if "candidate" in name else "reference")
                    return object()

                return types.SimpleNamespace(PreparedDynamicCSA=construct)

            namespace["importlib"] = types.SimpleNamespace(import_module=import_module)
            exec(binding, namespace)
            self.assertEqual(tuple(trace["package_mount"]), sides)
            self.assertEqual(tuple(trace["binding"]), sides)
            self.assertEqual(tuple(observed), sides)
            self.assertEqual(tuple(namespace["entries"]), sides)

    def eager_execution(self, order, candidate_value):
        block = source_block(
            NEW, '            eager_snapshots, tables, report["checks"]', "            if args.service_validation:"
        )
        with tempfile.TemporaryDirectory(prefix="csa_crossover_cpu_") as tmp:
            root, calls, comparisons = Path(tmp), [], []
            current = {}

            def entry(side):
                def invoke(**kwargs):
                    calls.append(("entry", side))
                    current["side"] = side
                    folder = root / side
                    folder.mkdir()
                    (folder / "kernel_config.py").write_text("CPU fake")

                return invoke

            def snapshot(side):
                self.assertEqual(current["side"], side)
                calls.append(("snapshot", side))
                return {"output": candidate_value if side == "candidate" else 7}, {"guard": True}

            def compare(actual, expected):
                comparisons.append((actual["output"], expected["output"]))
                return {"output": {"status": "PASS" if actual == expected else "FAIL"}}

            report = {"preparation_trace": {"compile_and_eager": []}}
            namespace = {
                "Path": Path,
                "os": types.SimpleNamespace(environ={"PYPTO_PROG_BUILD_DIR": str(root)}),
                "entries": {side: entry(side) for side in order_sides(order)},
                "report": report,
                "fixture": object(),
                "restore": lambda fixture: calls.append(("restore",)),
                "options": object(),
                "torch": types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda: None)),
                "snapshot": snapshot,
                "compare": compare,
                "save": lambda stage=None: None,
            }
            error = None
            try:
                exec(block, namespace)
            except AssertionError as exc:
                error = exc
            return calls, comparisons, report, error

    def test_eager_golden_is_reference_even_when_candidate_runs_first(self):
        for order in ("candidate-first", "reference-first"):
            calls, comparisons, report, error = self.eager_execution(order, 7)
            self.assertIsNone(error)
            self.assertEqual([event[1] for event in calls if event[0] == "entry"], list(order_sides(order)))
            self.assertEqual(comparisons, [(7, 7), (7, 7)])
            self.assertEqual(report["preparation_trace"]["compile_and_eager"], list(order_sides(order)))

    def test_wrong_candidate_never_becomes_golden(self):
        calls, comparisons, report, error = self.eager_execution("candidate-first", 99)
        self.assertIsNotNone(error)
        self.assertEqual([event[1] for event in calls if event[0] == "snapshot"], ["candidate", "reference"])
        self.assertEqual(comparisons, [(7, 7), (99, 7)])
        self.assertEqual(report["checks"]["reference_eager"]["numerical"]["output"]["status"], "PASS")
        self.assertEqual(report["checks"]["candidate_eager"]["numerical"]["output"]["status"], "FAIL")

    def test_graph_capture_changes_only_preparation_trace(self):
        def graph_loop(tree):
            return find(
                tree,
                lambda n: isinstance(n, ast.For)
                and ast.unparse(n.target) == "(side, entry)"
                and any(isinstance(x, ast.Call) and ast.unparse(x.func) == "torch.npu.NPUGraph" for x in ast.walk(n)),
            )

        new = copy.deepcopy(graph_loop(NEW_TREE))
        new.body = [
            node
            for node in new.body
            if not (isinstance(node, ast.Expr) and "report['preparation_trace']['graph_capture']" in ast.unparse(node))
        ]
        self.assertEqual(ast.dump(graph_loop(OLD_TREE)), ast.dump(new))

    def test_actual_sampling_loop_has_twenty_complete_windows(self):
        for order in ("candidate-first", "reference-first"):
            for layout in ("separate", "fused"):
                events, active, records = [], {"value": False}, []

                def restore(fixture, active=active, events=events):
                    self.assertFalse(active["value"])
                    events.append("restore")

                def begin(active=active, events=events):
                    self.assertFalse(active["value"])
                    active["value"] = True
                    events.append("begin")

                def end(active=active, events=events):
                    self.assertTrue(active["value"])
                    active["value"] = False
                    events.append("end")

                def graph(side, active=active, events=events):
                    def replay():
                        self.assertTrue(active["value"])
                        events.append(side)

                    return types.SimpleNamespace(replay=replay)

                def export(source, destination, side, table, *, init_layout, active=active):
                    self.assertFalse(active["value"])
                    return {"side": side, "init_layout": init_layout, "source": str(source)}

                namespace = {
                    "graphs": {side: graph(side) for side in order_sides(order)},
                    "records": records,
                    "fixture": object(),
                    "restore": restore,
                    "stream": types.SimpleNamespace(synchronize=lambda: None),
                    "pypto": types.SimpleNamespace(torch=types.SimpleNamespace(begin_dfx=begin, end_dfx=end)),
                    "args": types.SimpleNamespace(output=Path("run"), candidate_init_layout=layout),
                    "tables": {side: Path(side) for side in order_sides(order)},
                    "export_l1": export,
                }
                execute_node(sample_loop(NEW_TREE, 10), namespace)
                expected = [
                    side
                    for sample in range(10)
                    for side in (order_sides(order) if sample % 2 == 0 else tuple(reversed(order_sides(order))))
                ]
                self.assertEqual([record["side"] for record in records], expected)
                self.assertEqual(events, [event for side in expected for event in ("restore", "begin", side, "end")])
                for record in records:
                    self.assertEqual(record["init_layout"], layout if record["side"] == "candidate" else "separate")

    def test_launcher_requires_task_submit_and_explicit_valid_layout(self):
        launcher = HERE / "launch_crossover.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        source = launcher.read_text()
        self.assertIn("--shared-toolchain", source)
        self.assertIn('"$here/frozen_v3/run_pto.py"', source)
        self.assertIn('--prepare-order "$prepare_order"', source)
        self.assertIn("${TASK_DEVICE:?通过task-submit提交}", source)
        for layout in ("separate", "fused", "invalid"):
            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    str(HERE / "candidates/score_cross_step_v1"),
                    "h131072_b16_s6",
                    "cpu_reject_no_device",
                    layout,
                ],
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((HERE / "runs/h131072_b16_s6_cpu_reject_no_device").exists())


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CrossoverHarnessTest)
    )
    report = {
        "tests_run": result.testsRun,
        "passed": result.wasSuccessful(),
        "device_execution": False,
        "actual_AST_executed_with_CPU_fakes": [
            "package mount",
            "binding",
            "eager snapshots and reference golden",
            "ten-window sampling",
        ],
        "exact_AST_preserved": [
            "fixture",
            "snapshot",
            "compare",
            "restore calls",
            "warmup",
            "sampling",
            "post-L1 checks",
        ],
        "negative_control": "candidate-first wrong output fails against frozen reference after both snapshots",
        "limitations": "CPU contracts do not validate device graph capture or performance.",
    }
    (HERE / "crossover_harness_v3_contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
