# SPDX-License-Identifier: Apache-2.0
"""Trial: infer tensor capacity shapes while retaining runtime TND metadata.

Only annotations with dynamic capacity axes become bare pl.Tensor annotations.
Fixed model dimensions, NZ weight layouts, control flow and all math stay intact.
This uses the ordinary JIT tensor-metadata specialization; it does not generate
shape-specific algorithms or assume equal query lengths.
"""

import argparse
import ast
import difflib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-cache-dynamic", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    original = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    version = "v2" if args.keep_cache_dynamic else "v1"
    destination = here / f"runs/static_capacity_candidate_{version}"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dynamic_names = set()
    for path in original.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "dynamic"
            ):
                dynamic_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    changes, patch = {}, []
    for path in destination.glob("*.py"):
        before = path.read_text()
        tree = ast.parse(before)
        annotation_nodes = set()
        retained_cache_annotations = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.annotation:
                annotation_nodes.update(id(item) for item in ast.walk(node.annotation))
                if args.keep_cache_dynamic and node.arg == "idx_cache":
                    retained_cache_annotations.update(id(item) for item in ast.walk(node.annotation))
            if isinstance(node, ast.FunctionDef) and node.returns:
                annotation_nodes.update(id(item) for item in ast.walk(node.returns))
        # No body may rely directly on a removed DynVar binding.
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in dynamic_names
            and id(node) not in annotation_nodes
        ], path
        # AST column offsets count UTF-8 bytes, including Chinese comments.
        encoded = before.encode()
        starts = [0]
        for line in encoded.splitlines(keepends=True):
            starts.append(starts[-1] + len(line))
        replacements = []
        for node in ast.walk(tree):
            if (
                id(node) in annotation_nodes
                and id(node) not in retained_cache_annotations
                and isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pl"
                and node.value.attr == "Tensor"
                and any(
                    isinstance(item, ast.Name) and item.id in dynamic_names for item in ast.walk(node.slice)
                )
            ):
                replacements.append(
                    (starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset)
                )
        for start, end in sorted(replacements, reverse=True):
            encoded = encoded[:start] + b"pl.Tensor" + encoded[end:]
        after = encoded.decode()
        ast.parse(after)
        if after != before:
            path.write_text(after)
            changes[path.name] = len(replacements)
            patch.extend(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile="a/" + path.name,
                    tofile="b/" + path.name,
                )
            )
    (here / f"static_capacity_{version}.patch").write_text("".join(patch))
    (here / f"static_capacity_{version}_manifest.json").write_text(
        json.dumps(
            {
                "candidate": str(destination),
                "capacity_annotations_replaced": changes,
                "tnd_metadata": "Q/L, positions, page tables and active request geometry remain runtime tensors",
                "tradeoff": "One compiled variant per full tensor capacity signature and policy",
                "adopted": False,
                "indexer_cache_capacity_dynamic": args.keep_cache_dynamic,
            },
            indent=2,
        )
        + "\n"
    )
    print(destination, "annotations", sum(changes.values()))


if __name__ == "__main__":
    main()
