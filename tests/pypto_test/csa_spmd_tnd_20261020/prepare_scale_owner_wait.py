# SPDX-License-Identifier: Apache-2.0
"""Trial: acquire only the actual Native scale writers from the current plan.

The metadata planner rewrites all 48 list heads on every invocation. A nonempty
head identifies a writer; the same page owner serializes all updates to its
64-byte scale row. Each writer still acquires every compact-key producer before
committing scales and publishing its existing phase. Thus an acquired nonempty
writer covers both key completion and its scale updates. If every head is empty,
the key-store predicates imply that no Native indexer cache row is written.

No producer, arithmetic, slot, allocation, generation or FIFO handshake changes.
Consumers must already have acquired META; both current schedules satisfy this.
The unconditional final payload acquire is kept, including an empty writer set.
"""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

DEPENDENCY_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Acquire current-invocation scale writers after the caller acquired META."""

import pypto.language as pl

from .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_plan_stages import indexer_scale_plan_base
from .phase_sync import SIGNAL_SLOT_WORDS, SIGNAL_WORDS
from .runtime_abi import CORES, GROUPS, PHASE_IDX_SCALE
from .signals import csa_signal_wait


@pl.jit.inline(auto_scope=False)
def wait_indexer_scale_writers(
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    workspace: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    WORKERS: pl.constexpr,
):
    """The existing per-page plan includes every possible Native key write."""
    base = indexer_scale_plan_base(requests, tokens)
    for owner in pl.range(WORKERS):
        # All heads are rewritten by the unique META producer each invocation.
        # No signal slot or producer publication is elided or reused early.
        if pl.read(workspace, [base + owner]) >= 0:
            core = GROUPS + owner
            offset = (PHASE_IDX_SCALE * CORES + core) * SIGNAL_SLOT_WORDS
            signal = csa_signal_wait(signal, offset, generation)
    # Keep the business-data acquire even when no scale/key writer is active.
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal
'''


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    source = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    destination = here / "runs/scale_owner_wait_native_candidate_v1"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {}
    patches = []
    marker = (
        "    signal = wait_row_owners(signal, PHASE_IDX_SCALE, CORES, generation, "
        "0, VECTORS, 1, VECTORS, GROUPS, 1)"
    )
    for name in ("cube_schedule.py", "vector_schedule.py"):
        path = destination / name
        before = path.read_text()
        assert before.count(marker) == 1
        after = before.replace(
            "import pypto.language as pl\n", "import pypto.language as pl\n\n"
            "from .scale_dependencies import wait_indexer_scale_writers\n", 1,
        )
        after = after.replace(marker, "\n".join([
            "    signal = wait_indexer_scale_writers(",
            "        signal, group_offsets, tokens, pl.tensor.dim(idx_lengths, 0), generation, VECTORS",
            "    )",
        ]), 1)
        ast.parse(after)
        path.write_text(after)
        changes[name] = {
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        }
        patches.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    (destination / "scale_dependencies.py").write_text(DEPENDENCY_SOURCE)
    patches.extend(difflib.unified_diff(
        [], DEPENDENCY_SOURCE.splitlines(keepends=True),
        fromfile="/dev/null", tofile="b/scale_dependencies.py",
    ))
    unchanged = {}
    for path in source.glob("*.py"):
        if path.name not in changes:
            assert path.read_bytes() == (destination / path.name).read_bytes()
            unchanged[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "source": str(source), "candidate": str(destination), "adopted": False,
        "changes": changes, "unchanged_files": unchanged,
        "cube_meta_acquire": "IDX_QUERY_QUANT producer previously acquires META via compressor helper",
        "vector_meta_acquire": "indexer_compressor_vector_stage calls wait_producer(PHASE_META) unconditionally",
        "key_completion": "Every active scale writer still acquires PHASE_IDX_QUANT from all request owners",
        "empty_set": "Every legal Native key store implies a planned scale job; zero jobs means zero Native writes",
        "payload_acquire": "Unconditional fence/cacheinvalid/fence after the selected signal waits",
        "not_claimed": "CPU source/dependency audit is not device correctness or a performance result",
    }
    (here / "scale_owner_wait_v1.json").write_text(json.dumps(record, indent=2) + "\n")
    (here / "scale_owner_wait_v1.patch").write_text("".join(patches))
    print(destination)


if __name__ == "__main__":
    main()
