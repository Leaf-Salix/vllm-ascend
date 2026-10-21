# SPDX-License-Identifier: Apache-2.0
"""Address equivalence for existing proven layouts and unchanged generic fallback."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_key_page_proof import address_replacement, score_source


@pytest.mark.parametrize("direction", (-1, 0, 1))
@pytest.mark.parametrize("pages", (1, 2, 31, 32, 33, 1024, 1025))
def test_actual_address_expression_matches_original_page_lookup(direction, pages):
    first = pages + 17
    row = [first + direction * p for p in range(pages)]
    if direction == 0:
        # Includes duplicate/shared pages and non-monotonic mappings.
        row = [((p * 23) ^ (p // 2)) % (2 * pages + 19) for p in range(pages)]
    table = [row]
    reads = []

    def read(tensor, index):
        reads.append(tuple(index))
        return tensor[index[0]][index[1]]

    namespace = {
        "pl": SimpleNamespace(cast=lambda a, _: a, read=read, INDEX=None),
        "SCALE_PAGE_COALESCING": True,
        "key_page_direction": direction,
        "key_first_page": first,
        "idx_block_table": table,
        "buf_batch_idx": 0,
    }
    for target, safe in (("buf_prime_physical_page", "buf_prime_safe_page"), ("buf_physical_page", "buf_safe_page")):
        code = compile(address_replacement(target, safe), "actual_page_address", "exec")
        for logical in range(pages + 32):
            clamped = min(logical, pages - 1)
            namespace[safe] = clamped
            before = len(reads)
            exec(code, namespace)
            assert namespace[target] == row[clamped]
            assert len(reads) - before == (direction == 0)
        # A disabled plan must never consume stale first/direction words.
        namespace["SCALE_PAGE_COALESCING"] = False
        exec(code, namespace)
        assert namespace[target] == row[-1]
        namespace["SCALE_PAGE_COALESCING"] = True


def test_only_split_score_and_its_wrapper_change():
    reference = Path(__file__).resolve().parent / "runs/frozen_v122/indexer_score_stages.py"
    original = reference.read_text()
    revised = score_source(original)
    before = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef)
    }
    after = {
        n.name: ast.dump(n, include_attributes=False) for n in ast.parse(revised).body if isinstance(n, ast.FunctionDef)
    }
    assert before.keys() == after.keys()
    assert [name for name in before if before[name] != after[name]] == [
        "_indexer_score_cube_split_store",
        "indexer_score_cube_stage",
    ]

    def protocol_and_math(text):
        return [
            ast.dump(n, include_attributes=False)
            for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.Call)
            and ast.unparse(n.func)
            in (
                "pl.system.sync_wait",
                "pl.system.sync_set",
                "pl.matmul",
                "pl.gather_row",
                "pl.store",
            )
        ]

    assert protocol_and_math(original) == protocol_and_math(revised)
