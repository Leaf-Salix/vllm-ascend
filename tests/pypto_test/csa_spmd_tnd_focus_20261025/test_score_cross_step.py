# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for isolated Key-L1 carry; no SDK import or device execution."""

import ast
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
CANDIDATE = HERE / "candidates/score_cross_step_v1"


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def calls(node, names):
    return [
        ast.dump(call, include_attributes=False)
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and ast.unparse(call.func) in names
    ]


def simulate(steps, carry):
    """Track L1 last reader, Right QK ownership, page loads, and FIFO generations."""
    l1 = [None, None]
    l1_read = [True, True]
    right = [None, None]
    right_read = [True, True]
    gm, products, ready, ack = [], [], [], []

    def load(panel):
        slot = panel % 2
        assert l1_read[slot], (panel, "overwrite before MTE1 read", l1)
        l1[slot] = panel
        l1_read[slot] = False
        gm.extend((panel * 2, panel * 2 + 1))

    def extract(panel):
        slot = panel % 2
        assert l1[slot] == panel, (panel, "wrong L1 generation", l1)
        assert right_read[slot], (panel, "overwrite before QK read", right)
        l1_read[slot] = True
        right[slot] = panel
        right_read[slot] = False

    for step in range(steps):
        if step >= 2:
            ack.append(step - 2)
        first = step * 16
        if not carry or step == 0:
            load(first)
        extract(first)
        for local in range(16):
            panel = first + local
            if local < 15:
                load(panel + 1)
                extract(panel + 1)
            elif carry and step + 1 < steps:
                load(panel + 1)
            assert right[panel % 2] == panel
            right_read[panel % 2] = True
            products += [("QK0", panel), ("FIX0", panel), ("QK1", panel), ("FIX1", panel)]
            if local:
                products += [("WS0", panel - 1), ("WS1", panel - 1), ("STORE", panel - 1)]
        products += [("WS0", first + 15), ("WS1", first + 15), ("STORE", first + 15)]
        ready.append(step)
    ack.extend(range(max(steps - 2, 0), steps))
    assert all(l1_read) and all(right_read)
    assert ack == ready
    return gm, products, ready, ack


class ScoreCrossStepContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = functions(PARENT / "indexer_score_stages.py")
        cls.new = functions(CANDIDATE / "indexer_score_stages.py")
        cls.original = cls.old["_indexer_score_cube_split_store"]
        cls.carry = cls.new["_indexer_score_cube_cross_step"]

    def test_existing_helpers_unchanged(self):
        for name, node in self.old.items():
            if name != "indexer_score_cube_stage":
                self.assertEqual(ast.dump(node), ast.dump(self.new[name]), name)

    def test_live_half_unchanged(self):
        branches = []
        for function in (self.original, self.carry):
            branch = next(
                node
                for node in ast.walk(function)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "buf_live <= SPLIT_QK_ROWS // IDX_N_HEADS"
            )
            branches.append([ast.dump(node) for node in branch.body])
        self.assertEqual(*branches)

    def test_original_math_and_protocol_calls_unchanged(self):
        names = {"pl.tile.matmul_acc", "pl.tile.assemble", "pl.store", "pl.system.sync_set", "pl.system.sync_wait"}
        self.assertEqual(calls(self.original, names), calls(self.carry, names))

    def test_page_address_expressions_and_guards_unchanged(self):
        def address_ast(function):
            result = []
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name)
                    and t.id
                    in {
                        "buf_prime_key_row",
                        "buf_prime_safe_page",
                        "buf_prime_physical_page",
                        "buf_prime_native_byte",
                        "buf_key_row",
                        "buf_safe_page",
                        "buf_physical_page",
                        "buf_native_byte",
                    }
                    for t in node.targets
                ):
                    result.append(ast.dump(node))
                if isinstance(node, ast.If) and any(
                    word in ast.unparse(node.test)
                    for word in (
                        "buf_prime_safe_page",
                        "buf_prime_physical_page",
                        "buf_prime_native_byte",
                        "buf_safe_page",
                        "buf_physical_page",
                        "buf_native_byte",
                    )
                ):
                    result.append(ast.dump(node.test))
            return sorted(result)

        self.assertEqual(address_ast(self.original), address_ast(self.carry))
        self.assertEqual(sorted(calls(self.original, {"pl.gather_row"})), sorted(calls(self.carry, {"pl.gather_row"})))

    def test_policy_and_scope(self):
        policy = (PARENT / "policy.py").read_text()
        self.assertIn('"TO_L0": True', policy)
        self.assertIn("prefetch=1 if long_history or short_key_prefetch else 0", policy)
        wrapper = ast.unparse(self.new["indexer_score_cube_stage"])
        self.assertIn(
            "SPLIT_SCORE_ROWS and balance_leaves and (query_group_size == 6) "
            "and (qk_panel_cols == 64) and (score_tile == 1024)",
            wrapper,
        )
        self.assertIn("key_prefetch_panels == 1 and key_prefetch_to_l0", wrapper)

    def test_slot_ownership_and_protocol_for_all_leaf_steps(self):
        for steps in range(1, 9):
            with self.subTest(steps=steps):
                self.assertEqual(simulate(steps, False), simulate(steps, True))

    def test_tail_fragmented_reverse_and_invalid_page_traces(self):
        configurations = 0
        for steps in range(1, 9):
            old, new = simulate(steps, False)[0], simulate(steps, True)[0]
            for cache_len in (0, 1, 31, 32, 33, 1023, 1024, 1025, steps * 1024 - 1, steps * 1024):
                for page_begin in (0, 32, 224):
                    for table in (
                        list(range(300)),
                        list(reversed(range(300))),
                        [(i * 17) % 300 for i in range(300)],
                        [-1, 301, 0, 2],
                    ):

                        def addresses(trace, page_begin=page_begin, cache_len=cache_len, table=table):
                            result = []
                            for page in trace:
                                safe = min(page_begin + page, max((cache_len - 1) // 32, 0))
                                if 0 <= safe < len(table) and 0 <= table[safe] < 300:
                                    byte = table[safe] * 4160
                                    result.append((safe, byte // 128, byte % 128))
                            return result

                        self.assertEqual(addresses(old), addresses(new))
                        configurations += 1
        self.assertEqual(configurations, 960)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScoreCrossStepContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "candidate": "score_cross_step_v1",
        "tests_run": result.testsRun,
        "passed": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "ownership_steps": list(range(1, 9)),
        "address_configurations": 960,
        "device_execution": False,
        "performance_eligible": False,
        "limitation": "Source and serial ownership model; generated asynchronous dependencies require separate audit.",
    }
    (HERE / "score_cross_step_v1_contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
