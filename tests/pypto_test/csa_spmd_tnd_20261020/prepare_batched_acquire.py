# SPDX-License-Identifier: Apache-2.0
"""Trial: acquire payload once after adjacent sets of producer waits.

The last wait in every selected sequence has a positive token-capacity count or
the fixed 48-vector count. It therefore retains the full fence/cache/fence after
all required signals. No payload read, arithmetic stage or branch is crossed.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path


def phase_wait(statement):
    if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if not isinstance(call.func, ast.Name) or call.func.id != "wait_row_owners":
        return None
    return call if isinstance(call.args[1], ast.Name) else None


def replace_calls(source, sequences):
    tree = ast.parse(source)
    calls = []
    found = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for index in range(len(function.body)):
            for sequence in sequences:
                statements = function.body[index : index + len(sequence)]
                matched = [phase_wait(statement) for statement in statements]
                if len(matched) != len(sequence) or any(call is None for call in matched):
                    continue
                if tuple(call.args[1].id for call in matched) == sequence:
                    calls.extend(matched[:-1])
                    found.append(sequence)
    assert sorted(found) == sorted(sequences), found
    encoded = source.encode()
    starts = [0]
    for line in encoded.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    replacements = []
    for call in calls:
        start = starts[call.lineno - 1] + call.col_offset
        end = starts[call.end_lineno - 1] + call.end_col_offset
        call.keywords.append(ast.keyword(arg="ACQUIRE", value=ast.Constant(value=False)))
        replacements.append((start, end, ast.unparse(call).encode()))
    for start, end, replacement in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode(), len(calls)


def bind_remaining_acquires(source):
    """JIT dependency calls require explicit constexpr bindings, even with defaults."""
    encoded = source.encode()
    starts = [0]
    for line in encoded.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    replacements = []
    for call in ast.walk(ast.parse(source)):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "wait_row_owners" or any(k.arg == "ACQUIRE" for k in call.keywords):
            continue
        start = starts[call.lineno - 1] + call.col_offset
        end = starts[call.end_lineno - 1] + call.end_col_offset
        call.keywords.append(ast.keyword(arg="ACQUIRE", value=ast.Constant(value=True)))
        replacements.append((start, end, ast.unparse(call).encode()))
    for start, end, replacement in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode()


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/native_wait_candidate_v1"
    destination = here / "runs/batched_acquire_native_candidate_v2"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}
    path = destination / "phase_sync.py"
    before = path.read_text()
    marker = "    producer_stride: pl.Scalar[pl.INDEX],\n):"
    assert before.count(marker) == 1
    source = before.replace(marker, "    producer_stride: pl.Scalar[pl.INDEX],\n    ACQUIRE: pl.constexpr = True,\n):")
    start = source.index("def wait_row_owners(")
    prefix, body = source[:start], source[start:]
    old = "        pl.system.fence()\n        pl.system.cacheinvalid()\n        pl.system.fence()\n"
    assert body.count(old) == 1
    body = body.replace(old, "        if ACQUIRE:\n" + "".join("    " + line for line in old.splitlines(keepends=True)))
    path.write_text(prefix + body)
    changes[path.name] = (before, prefix + body)
    selections = {
        "vector_schedule.py": [
            ("PHASE_ROPE", "PHASE_QR"),
            ("PHASE_WEIGHTS", "PHASE_IDX_QUERY_QUANT"),
            ("PHASE_ORIGINAL_CACHE", "PHASE_MAIN_OUTPUT", "PHASE_ROPE"),
        ],
        "cube_schedule.py": [("PHASE_COEFF", "PHASE_IDX_SCALE")],
    }
    counts = {}
    for name, sequences in selections.items():
        path = destination / name
        before = path.read_text()
        after, counts[name] = replace_calls(before, sequences)
        path.write_text(after)
        changes[name] = (before, after)
    for path in destination.glob("*.py"):
        before = path.read_text()
        after = bind_remaining_acquires(before)
        if before != after:
            path.write_text(after)
            changes[path.name] = (changes.get(path.name, (before,))[0], after)
    patch = []
    for name, (before, after) in changes.items():
        ast.parse(after)
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                         fromfile="a/" + name, tofile="b/" + name))
    (here / "batched_acquire_v2.patch").write_text("".join(patch))
    (here / "batched_acquire_v2.json").write_text(json.dumps({
        "candidate": str(destination), "acquires_coalesced_per_role": counts,
        "retained": "Every signal wait, every publisher, and one final payload acquire per selected sequence",
        "guard": "The final wait always has positive capacity or a fixed participant count",
        "adopted": False,
    }, indent=2) + "\n")
    print(destination, counts)


if __name__ == "__main__":
    main()
