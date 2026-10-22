# SPDX-License-Identifier: Apache-2.0
"""Wait only for the Q_B column writers consumed by each AIV's head group."""

import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

from prepare_control_candidates import replace_once

HELPER = '''# SPDX-License-Identifier: Apache-2.0
"""Q_B whole-phase waits for the exact column owners consumed by an AIV.

Every Q_B owner produces all token rows of its columns before publishing.
The dequant head-group count is 1/2/4/8/16, all divisors of 48, so a given
AIV keeps the same head group across its token waves. Do not substitute an
old column count for the wide-N branch or a logical rank for a physical core.
All publications and the generation/FIFO protocols remain unchanged.
"""

import pypto.language as pl

from .phase_sync import SIGNAL_SLOT_WORDS, SIGNAL_WORDS
from .query_projection_stages import select_dequant_head_tile
from .runtime_abi import CORES, GROUPS, PHASE_QB, VECTORS
from .signals import csa_signal_wait


@pl.jit.inline(auto_scope=False)
def wait_query_projection_columns(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    COOPERATIVE_PREFIX: pl.constexpr,
    PREFIX_BRANCH_GROUPS: pl.constexpr,
    QB_WIDE_N: pl.constexpr,
):
    head_tile = select_dequant_head_tile(tokens)
    head_groups = 64 // head_tile
    tasks = ((tokens + 7) // 8) * head_groups
    if worker < tasks:
        column_width = 256
        writers = GROUPS
        if COOPERATIVE_PREFIX:
            writers = GROUPS - PREFIX_BRANCH_GROUPS
            if QB_WIDE_N:
                column_width = 512
        first_column = (worker % head_groups) * head_tile * 512 // column_width
        columns = head_tile * 512 // column_width
        for ordinal in pl.range(pl.min(columns, writers)):
            logical_owner = (first_column + ordinal) % writers
            physical_owner = logical_owner
            if COOPERATIVE_PREFIX:
                physical_owner = (
                    (logical_owner + writers - PREFIX_BRANCH_GROUPS % writers) % writers + PREFIX_BRANCH_GROUPS
                )
            offset = (PHASE_QB * CORES + physical_owner) * SIGNAL_SLOT_WORDS
            signals = csa_signal_wait(signals, offset, generation)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
    return signals
'''


def factor_head_selector(source):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    start = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "dq_token_tiles")
    condition = next(n for n in fn.body if isinstance(n, ast.If))
    lines = source.splitlines(keepends=True)
    old_block = "".join(lines[start.lineno - 1 : condition.end_lineno])
    helper = (
        "@pl.jit.inline(auto_scope=False)\ndef select_dequant_head_tile(tokens: pl.Scalar[pl.INDEX]):\n"
        + old_block
        + "    return dq_head_tile\n\n\n"
    )
    lines[start.lineno - 1 : condition.end_lineno] = ["    dq_head_tile = select_dequant_head_tile(tokens)\n"]
    changed = "".join(lines)
    marker = "@pl.jit.inline(auto_scope=False)\ndef q_dequant_rope_stage("
    changed = replace_once(changed, marker, helper + marker)
    return changed


def append_parameter(source, function, name, default):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = source.splitlines(keepends=True)
    lines.insert(fn.args.args[-1].end_lineno, f"    {name}: pl.constexpr = {default},\n")
    return "".join(lines)


def append_arguments(source, function, arguments):
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and ast.unparse(n.func) == function]
    assert len(calls) == 1
    call = calls[0]
    lines = source.splitlines(keepends=True)
    close = lines[call.end_lineno - 1]
    assert close.strip() == ")"
    indent = close[: len(close) - len(close.lstrip())] + "    "
    lines[call.end_lineno - 1 : call.end_lineno - 1] = [indent + arg + ",\n" for arg in arguments]
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    vector = (reference / "vector_schedule.py").read_text()
    vector = replace_once(
        vector,
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .query_projection_dependencies import wait_query_projection_columns\n",
    )
    vector = replace_once(
        vector,
        "    signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, 128, 1, GROUPS, 0, 1)\n",
        "    signal = wait_query_projection_columns(\n"
        "        signal, tokens, worker, generation, COOPERATIVE_PREFIX, PREFIX_BRANCH_GROUPS, QB_WIDE_N\n    )\n",
    )
    vector = append_parameter(vector, "vector_schedule", "PREFIX_BRANCH_GROUPS", "4")
    vector = append_parameter(vector, "vector_schedule", "QB_WIDE_N", "False")
    entry = append_arguments(
        (reference / "entry.py").read_text(), "vector_schedule", ["PREFIX_BRANCH_GROUPS", "QB_WIDE_N"]
    )
    revised = {
        "query_projection_stages.py": factor_head_selector((reference / "query_projection_stages.py").read_text()),
        "query_projection_dependencies.py": textwrap.dedent(HELPER),
        "vector_schedule.py": vector,
        "entry.py": entry,
    }
    target = here / "runs/query_exact_waits"
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, source in revised.items():
        ast.parse(source)
        (target / name).write_text(source)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "changed_files": list(revised),
        "scope": "Only the Q_B-to-dequant producer wait set changes; same head selector and original math",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / "query_exact_waits.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
