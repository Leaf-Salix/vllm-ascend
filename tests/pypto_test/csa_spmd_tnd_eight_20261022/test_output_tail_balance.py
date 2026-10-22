# SPDX-License-Identifier: Apache-2.0
"""Execute the real output rank selection and verify full coverage and quant dependencies."""

import ast
import collections
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def root():
    path = Path(__file__).with_name("runs") / "output_tail_balance_v1"
    if not path.is_dir():
        pytest.skip("Materialize the output scheduling candidate first")
    return path


@pytest.mark.parametrize("tokens", [129, 130, 143, 144, 145, 159, 160])
def test_all_64_jobs_exactly_once_after_correct_group_quant(tokens):
    tree = ast.parse((root() / "cube_schedule.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_TAIL_BALANCE")
    program = compile(ast.Module(body=branch.body, type_ignores=[]), "actual_output_mapping", "exec")
    owned = []
    for worker in range(24):
        waits = []

        def wait(signal, phase, cores, generation, begin, count, step, vectors, offset, factor, *, waits=waits):
            group = phase - 100
            assert begin == group * ((tokens + 31) // 32) and count == (tokens + 31) // 32
            assert (cores, generation, step, vectors, offset, factor) == (72, 1, 1, 48, 24, 1)
            waits.append(group)
            return signal

        def compute(quant, weight, output, rows, rank, workers, tile, begin, end, *, waits=waits, owner=worker):
            assert rows == tokens and workers == 64 and tile == 128 and end == begin + 1
            assert waits.pop() == begin
            # Original NZ helper's integer work enumeration; mathematical body is unchanged.
            for wave in range(begin * 8 // workers, (end * 8 - rank + workers - 1) // workers):
                work = rank + wave * workers
                if work >= begin * 8:
                    owned.append((work, owner))
            return output

        values = dict(
            worker=worker,
            pl=SimpleNamespace(range=range),
            GROUPS=24,
            CORES=72,
            VECTORS=48,
            PHASE_O_QUANT_GROUP_BASE=100,
            generation=1,
            signal=None,
            quant_per_group=(tokens + 31) // 32,
            wait_row_owners=wait,
            o_b_groups_stage=compute,
            o_quant=None,
            wo_b=None,
            o_partials=None,
            tokens=tokens,
            OB_ROWS=128,
        )
        exec(program, values)
        assert not waits
    assert sorted(job for job, _ in owned) == list(range(64))
    counts = collections.Counter(worker for _, worker in owned)
    assert [counts[w] for w in range(24)] == [2] * 4 + [3] * 4 + [2] * 4 + [3] * 4 + [2] * 4 + [4] * 4
    changed = [(job, worker) for job, worker in owned if worker != job % 24]
    assert sorted(job for job, _ in changed) == list(range(4)) + list(range(8, 12))


@pytest.mark.parametrize("tokens", [96, 127, 128, 129, 144, 160, 161, 168, 192, 256])
def test_scope_and_preserved_math(tokens):
    candidate = root()
    here = Path(__file__).resolve().parent
    old = runpy.run_path(str(here / "runs/baseline/policy.py"))["select_policy"]
    new = runpy.run_path(str(candidate / "policy.py"))["select_policy"]
    for history in (8192, 131072, 262144):
        kwargs = dict(tokens=tokens, requests=(tokens + 5) // 6, max_query_tokens=6, max_seq_len=history + 6)
        actual = new(**kwargs).constexprs()
        assert actual.pop("OUTPUT_TAIL_BALANCE") == (history > 32768 and 128 < tokens <= 160)
        assert actual == old(**kwargs).constexprs()
    baseline = json.loads((here / "baseline.json").read_text())["files"]
    changes = {
        n for n, digest in baseline.items() if hashlib.sha256((candidate / n).read_bytes()).hexdigest() != digest
    }
    assert changes == {"policy.py", "entry.py", "cube_schedule.py"}
