# SPDX-License-Identifier: Apache-2.0
"""Materialize the active scale-owner list once in the unique META producer.

The first trial scanned 48 heads independently in all 72 Score consumers and
regressed. This variant moves that scan into the preexisting unique planner and
appends 64 INT32 words to its private workspace. Every invocation overwrites the
count and valid list prefix. Old layout-proof offsets and payload math remain
unchanged; no new GM root, notification or runtime metadata transfer is added.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path

DEPENDENCY_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Publish a private active-writer list inside the existing META phase."""

import pypto.language as pl

from .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_plan_stages import indexer_scale_plan_base
from .phase_sync import SIGNAL_SLOT_WORDS, SIGNAL_WORDS
from .runtime_abi import CORES, GROUPS, PHASE_IDX_SCALE
from .signals import csa_signal_wait

SCALE_WRITER_LIST_WORDS = 64


@pl.jit.inline(auto_scope=False)
def indexer_scale_writer_list_stage(
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """Only the unique META writer calls this, after filling all current heads."""
    heads = indexer_scale_plan_base(requests, tokens)
    owner_base = pl.tensor.dim(workspace, 0) - SCALE_WRITER_LIST_WORDS
    active = pl.cast(0, pl.INDEX)
    for owner in pl.range(WORKERS):
        if pl.read(workspace, [heads + owner]) >= 0:
            pl.write(workspace, [owner_base + 1 + active], pl.cast(owner, pl.INT32))
            active = active + 1
    # Zero active writers explicitly overwrite the count, preventing stale reuse.
    pl.write(workspace, [owner_base], pl.cast(active, pl.INT32))
    return workspace


@pl.jit.inline(auto_scope=False)
def wait_indexer_scale_writers(
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    workspace: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    WORKERS: pl.constexpr,
):
    """Caller acquired META; all producer publications and key waits are kept."""
    owner_base = pl.tensor.dim(workspace, 0) - SCALE_WRITER_LIST_WORDS
    count = pl.cast(pl.read(workspace, [owner_base]), pl.INDEX)
    for ordinal in pl.range(count):
        owner = pl.cast(pl.read(workspace, [owner_base + 1 + ordinal]), pl.INDEX)
        core = GROUPS + owner
        offset = (PHASE_IDX_SCALE * CORES + core) * SIGNAL_SLOT_WORDS
        signal = csa_signal_wait(signal, offset, generation)
    # The business acquire is still unconditional, including an empty list.
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal
'''


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/scale_owner_wait_native_candidate_v1"
    destination = here / "runs/scale_owner_list_native_candidate_v2"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []
    updates = {}
    path = destination / "binding.py"
    text = path.read_text()
    text = text.replace(
        "from .policy import CSASPMDPolicy, select_policy\n",
        "from .policy import CSASPMDPolicy, select_policy\n"
        "from .scale_dependencies import SCALE_WRITER_LIST_WORDS\n", 1,
    )
    marker = "+ (2 * b if self.policy.scale_page_coalescing else 0),),"
    assert text.count(marker) == 1
    updates[path.name] = text.replace(marker,
        "+ (2 * b if self.policy.scale_page_coalescing else 0) + SCALE_WRITER_LIST_WORDS,),", 1)
    path = destination / "vector_schedule.py"
    text = path.read_text().replace(
        "from .scale_dependencies import wait_indexer_scale_writers",
        "from .scale_dependencies import indexer_scale_writer_list_stage, wait_indexer_scale_writers", 1,
    )
    marker = "        if BALANCED and SCALE_PAGE_COALESCING:\n"
    assert text.count(marker) == 1
    updates[path.name] = text.replace(marker,
        "        group_offsets = indexer_scale_writer_list_stage(\n"
        "            group_offsets, tokens, pl.tensor.dim(idx_lengths, 0), VECTORS\n"
        "        )\n" + marker, 1)
    updates["scale_dependencies.py"] = DEPENDENCY_SOURCE
    for name, after in updates.items():
        path = destination / name
        before = path.read_text()
        ast.parse(after)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                           fromfile="v1/" + name, tofile="v2/" + name))
    record = {
        "candidate": str(destination), "adopted": False, "parent": str(source),
        "private_workspace_added_bytes": 64 * 4, "new_tensor_roots": 0,
        "old_offset_regions_changed": False, "planner": "Same unique META producer",
        "preserved": "Every signal publication, exact arithmetic, key/scale ownership, FFTS, final payload acquire",
        "dynamic": "Every invocation writes count and valid prefix; unused list entries are not read",
        "required_validation": (
            "Compile and inspect count loop; full exact/protocol/L1; zero-job S1 and changed metadata"
        ),
    }
    (here / "scale_owner_list_v2.patch").write_text("".join(patches))
    (here / "scale_owner_list_v2.json").write_text(json.dumps(record, indent=2) + "\n")
    # Reuse actual descriptors, changing only the private allocation's extent.
    original = here / "runs/cohort_prefix_a241_h131072_b16_s6_v41/buffer_descriptors.json"
    description = json.loads(original.read_text())
    description["buffers"]["group_offsets"]["shape"][0] += 64
    (here / "runs/scale_owner_list_b16_descriptors.json").write_text(json.dumps(description, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
