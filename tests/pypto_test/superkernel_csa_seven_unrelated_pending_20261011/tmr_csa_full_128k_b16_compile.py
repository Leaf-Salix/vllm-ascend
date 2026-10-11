# SPDX-License-Identifier: Apache-2.0
"""Compile the frozen baseline and private Scalar-overlap CSA without an NPU."""

import argparse
import importlib
import inspect
import json
from pathlib import Path

import torch
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.ir.compile import _compile_impl
from pypto.pypto_core.passes import RuntimeKind
from pypto.runtime import RunConfig

BATCH = 16
HISTORY = 131072
TOKENS = BATCH * 6
PACKAGES = {
    "baseline": "deepseek_v4_flash_csa_scalar_overlap_demo",
    "local": "deepseek_v4_flash_csa_local_boundaries",
    "all": "deepseek_v4_flash_csa_all_boundaries",
    "overlap": "deepseek_v4_flash_csa_scalar_overlap_full",
}
TRACE_PACKAGE = "deepseek_v4_flash_csa_scalar_overlap_trace"


def shape_only_args(root, batch=BATCH, history=HISTORY):
    """Use the Native single-layer fixture's page geometry; bind no device."""
    cache_columns = (history + 6 + 127) // 128 + 1
    dynamic = {
        "T_DYN": batch * 6,
        "B_DYN": batch,
        "ORI_BLOCK_NUM_DYN": batch * 9 + 1,
        "CMP_BLOCK_NUM_DYN": batch * cache_columns + 1,
        "IDX_NATIVE_CACHE_BLOCK_NUM_DYN": batch * cache_columns + 1,
        "CSA_STATE_BLOCK_NUM_DYN": batch * 9 + 1,
        "INNER_STATE_BLOCK_NUM_DYN": batch * 9 + 1,
        "CSA_STATE_PAGE_ELEMENTS": 4096,
        "CSA_INNER_STATE_PAGE_ELEMENTS": 1024,
        "CSA_STATE_TABLE_COLUMNS": (history + 7) // 2 + 1,
        "CSA_INNER_STATE_TABLE_COLUMNS": (history + 7) // 2 + 1,
        "CSA_ORIGINAL_TABLE_COLUMNS": (history + 37) // 32 + 1,
        "CSA_COMPRESSED_TABLE_COLUMNS": cache_columns,
        "CSA_INDEXER_TABLE_COLUMNS": cache_columns,
        "CSA_INDEXER_PAGE_BYTES": 32 * (128 + 2),
        "CSA_QUERY_BOUNDS": batch + 1,
        "CSA_COMPRESSED_METADATA_ROWS": batch,
        "CSA_INDEXER_METADATA_ROWS": batch,
    }
    dtypes = {
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    arguments, description = [], {}
    for name, parameter in inspect.signature(root).parameters.items():
        annotation = parameter.annotation
        if name == "host_max_seq_len":
            continue  # TMR's unused constexpr, not a runtime argument.
        shape = [dim if isinstance(dim, int) else dynamic[dim.name] for dim in annotation.shape]
        value = torch.empty(shape, dtype=dtypes[str(annotation.dtype)], device="meta")
        arguments.append(value)
        description[name] = {"shape": shape, "dtype": str(value.dtype)}
    return tuple(arguments), description


def compile_variant(name, output):
    package = TRACE_PACKAGE if name == "trace" else PACKAGES[name]
    module = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.decode_csa")
    kernel = module.decode_csa_tp1_layer_test
    kernel._get_dep_graph()
    arguments, description = shape_only_args(module._decode_csa_tp1_layer)
    program = kernel.specialize(*arguments)
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.py").write_text(str(program))
    (output / "input_shapes.json").write_text(json.dumps(description, indent=2) + "\n")
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer")
    options = RunConfig(platform="a2a3").compile_kwargs()
    options.update(output_dir=str(output / "build"), runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER)
    artifact = _compile_impl(program, _kernel_abi=abi, **options)
    artifact.load()
    print(f"COMPILED {name}: {output}; no device execution", flush=True)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=(*PACKAGES, "trace"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    for variant in (args.variant,) if args.variant else PACKAGES:
        compile_variant(variant, args.output / variant)


if __name__ == "__main__":
    main()
