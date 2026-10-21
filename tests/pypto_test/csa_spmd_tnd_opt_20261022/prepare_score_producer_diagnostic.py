# SPDX-License-Identifier: Apache-2.0
"""Create an isolated, unthrottled Score producer for instruction simulation.

This is NOT a full CSA candidate. It has one AIC, one synthetic G6 leaf of six
1024-column steps, no consumer and no inter-core FIFO. Only the FIFO calls and
their now-empty guards are removed from the copied helper. It must never enter
the nine-case L1 acceptance gate. The formal source and toolchain are untouched.
"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

BUFFERS = {
    "query": {"shape": [384, 128], "dtype": "torch.int8"},
    "coefficients": {"shape": [16, 384], "dtype": "torch.float16"},
    "key0": {"shape": [6272, 128], "dtype": "torch.int8"},
    "key64": {"shape": [6272, 128], "dtype": "torch.int8"},
    "cache": {"shape": [193, 4160], "dtype": "torch.int8"},
    "table": {"shape": [1, 192], "dtype": "torch.int32"},
    "positions": {"shape": [6], "dtype": "torch.int64"},
    "lengths": {"shape": [1], "dtype": "torch.int32"},
    "bounds": {"shape": [2], "dtype": "torch.int32"},
    "offsets": {"shape": [32], "dtype": "torch.int32"},
    "plan": {"shape": [8], "dtype": "torch.int32"},
    "transfer": {"shape": [12, 1024], "dtype": "torch.float32"},
}


class RemoveFifo(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in (
            "pl.system.sync_wait",
            "pl.system.sync_set",
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.body and not node.orelse:
            return None
        assert node.body
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if not node.body:
            assert not node.orelse
            return None
        return node


def isolate(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    original = ast.get_source_segment(source, fn)
    before_calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    fn = RemoveFifo().visit(fn)
    fn.decorator_list = []
    replacement = ast.unparse(ast.fix_missing_locations(fn))
    after_calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert before_calls.count("pl.system.sync_wait") == 2
    assert before_calls.count("pl.system.sync_set") == 1
    assert not any(n.startswith("pl.system.sync_") for n in after_calls)
    result = source.replace(original, replacement, 1)
    ast.parse(result)
    return result


def entry(panel, to_l0, revision):
    dtype = {
        "torch.int8": "INT8",
        "torch.float16": "FP16",
        "torch.int32": "INT32",
        "torch.int64": "INT64",
        "torch.float32": "FP32",
    }
    signature = []
    for name, row in BUFFERS.items():
        annotation = f"pl.Tensor[{row['shape']}, pl.{dtype[row['dtype']]}]"
        if name == "transfer":
            annotation = f"pl.InOut[{annotation}]"
        signature.append(f"    {name}: {annotation},")
    args = ", ".join(BUFFERS)
    signature_text = "\n".join(signature)
    source = f'''# SPDX-License-Identifier: Apache-2.0
"""SYNTHETIC PRODUCER DIAGNOSTIC: one AIC, no FIFO consumer, not CSA latency."""
import pypto.language as pl
from .indexer_score_stages import _indexer_score_cube_split_store

@pl.jit.incore
def score_producer(
{signature_text}
):
    worker = pl.tile.get_block_idx()
    transfer = _indexer_score_cube_split_store(
        {args}, worker, 6, {panel}, 1024, 1, {to_l0}, True, 0, 1,
    )
    return transfer

@pl.jit
def decode_csa_spmd(
{signature_text}
):
    with pl.cluster(1):
        transfer = score_producer({args})
    return transfer
'''
    if revision == 2:
        source = source.replace("    with pl.cluster(1):\n        transfer =", "    transfer =", 1)
    return source


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    suffix = "" if args.revision == 1 else "_v2"
    reports = []
    for label, parent_name, panel, to_l0 in (
        ("v122", "frozen_v122", 64, True),
        ("n256", "score_query_pipeline", 256, False),
    ):
        parent = here / "runs" / parent_name
        expected_name = "frozen_v122.json" if label == "v122" else "score_query_pipeline.json"
        expected = json.loads((here / expected_name).read_text())["files"]
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
        target = here / "runs" / f"producer_diagnostic_{label}{suffix}"
        shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (target / "indexer_score_stages.py").write_text(isolate((parent / "indexer_score_stages.py").read_text()))
        (target / "entry.py").write_text(entry(panel, to_l0, args.revision))
        descriptor = here / "descriptors" / f"producer_diagnostic_{label}{suffix}.json"
        with descriptor.open("x") as stream:
            stream.write(json.dumps({"buffers": BUFFERS, "policy": {}}, indent=2) + "\n")
        reports.append(
            dict(
                label=label,
                parent=str(parent),
                source=str(target),
                descriptor=str(descriptor),
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
            )
        )
    report = {
        "scope": "synthetic unthrottled single-AIC Score producer; not a 72-role CSA execution",
        "formal_gate_eligible": False,
        "hardware_execution": False,
        "removed_for_isolation": ["SCORE_READY notification", "SCORE_CONSUMED waits", "empty drain loop"],
        "controls": {
            "groups": 1,
            "leaves": 1,
            "leaf_tiles": 6,
            "extra_leaves": 0,
            "request_lengths": [24576],
            "query_bounds": [0, 6],
            "positions": list(range(24570, 24576)),
            "group_descriptor_at_offset_16": [0, 0, 6, 6144],
            "offsets_prefix": [0, 1],
            "block_table": "192..1 descending, 32 INT8 key rows per 4160-byte Native page",
            "query_and_keys": "all INT8 ones; no cache writes, synthetic separate aliases agree on every byte",
            "coefficients": "FP16 ones in row q, columns q*64..(q+1)*64, q=0..5; other entries zero",
        },
        "expected_work": {"producer_roles": 1, "groups": 1, "leaves": 1, "score_steps": 6},
        "cases": reports,
    }
    with (here / f"score_producer_diagnostic{suffix}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({r["label"]: r["source"] for r in reports}))


if __name__ == "__main__":
    main()
