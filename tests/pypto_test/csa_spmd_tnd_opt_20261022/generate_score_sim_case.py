# SPDX-License-Identifier: Apache-2.0
"""Extend the bundled profiling generator with exact Score inputs and ABI dimensions."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

from prepare_score_producer_diagnostic import BUFFERS


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def use_cann92_builtin_types(source):
    # Current CANN and PTO-ISA already define these types. Remove only the old
    # testcase preamble's fallback declarations; keep the kernel body intact.
    for name in ("hifloat8_t", "float8_e4m3_t", "float8_e5m2_t", "float8_e8m0_t", "float4_e1m2x2_t", "float4_e2m1x2_t"):
        line = "typedef struct { unsigned char v; } " + name + ";\n"
        assert source.count(line) == 1, name
        source = source.replace(line, "", 1)
    source, count = re.subn(
        r"#if !defined\(__CCE_AICORE__\) && !defined\(TMRGSORT_HPP\)\nnamespace pto \{.*?"
        r"\n\}(?: // namespace pto)?\n#endif\n",
        "",
        source,
        flags=re.DOTALL,
    )
    assert count == 1
    return source


def write_inputs(directory):
    import numpy as np

    directory = Path(directory)
    binding = json.loads((directory / "input_bindings.json").read_text())
    hashes = {}
    for row in binding["buffers"]:
        name = row["logical_name"]
        shape = BUFFERS[name]["shape"]
        dtype = getattr(np, BUFFERS[name]["dtype"].removeprefix("torch."))
        value = np.zeros(shape, dtype=dtype)
        if name in ("query", "key0", "key64", "cache"):
            value.fill(1)
        elif name == "coefficients":
            for q in range(6):
                value[q, q * 64 : (q + 1) * 64] = 1
        elif name == "table":
            value[0] = np.arange(192, 0, -1)
        elif name == "positions":
            value[:] = np.arange(24570, 24576)
        elif name == "lengths":
            value[:] = [24576]
        elif name == "bounds":
            value[:] = [0, 6]
        elif name == "offsets":
            value[:2] = [0, 1]
            value[16:20] = [0, 0, 6, 6144]
        elif name == "plan":
            value[:] = [1, 6144, 1, 6, 0, 0, 0, 0]
        elif name == "transfer":
            value.fill(-777.0)
        else:
            raise ValueError(name)
        assert value.size == row["elements"]
        path = directory / f"{row['cpp_name']}.bin"
        value.tofile(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


def bindings(source, gen):
    cpp = source.read_text()
    pto = source.with_suffix(".pto").read_text()
    name, mixed, params = gen.parse_cpp(cpp)
    assert name == "score_producer" and not mixed
    assert len(params) == 28 and all(p.is_ptr for p in params[:12]) and not any(p.is_ptr for p in params[12:])
    wrapper = source.parent.parent / "kernels/aic/score_producer.cpp"
    packed = wrapper.read_text()
    (call,) = re.findall(r"^\s+score_producer\(([^;]+)\);", packed, re.MULTILINE)
    arguments = [a.strip() for a in call.split(",")]
    assert len(arguments) == len(params)
    assert arguments[:12] == [f"{name}__ssa_v0" for name in BUFFERS]
    dims = {}
    for scalar, tensor, axis in re.findall(
        r"int64_t (\w+) = static_cast<int64_t>\((\w+)__ssa_v0_tensor->shapes\[(\d+)\]\);", packed
    ):
        dims[scalar] = BUFFERS[tensor]["shape"][int(axis)]
    dims.update(__pypto_spmd_block_idx=0, __pypto_spmd_block_num=1)
    scalar_values = {i: dims[arg] for i, arg in enumerate(arguments) if i >= 12}

    # Validate every source-root view against the same dimensions used for allocation.
    constants = {n: int(v) for n, v in re.findall(r"(%\w+) = arith.constant (-?\d+) : (?:index|i\d+)", pto)}
    constants.update({f"%arg{i}": v for i, v in scalar_values.items()})
    views = []
    for root, shapes, strides in re.findall(
        r"pto.make_tensor_view %arg(\d+), shape = \[([^]]+)\], strides = \[([^]]+)\]", pto
    ):
        idx = int(root)
        shape = [constants[v.strip()] for v in shapes.split(",")]
        stride = [constants[v.strip()] for v in strides.split(",")]
        logical = list(BUFFERS)[idx]
        assert shape == BUFFERS[logical]["shape"], (logical, shape)
        footprint = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
        assert footprint <= math.prod(shape)
        views.append(dict(logical_name=logical, shape=shape, strides=stride, elements=footprint))
    counts = {p.name: math.prod(row["shape"]) for p, row in zip(params[:12], BUFFERS.values())}
    expected_dtypes = (
        "int8_t",
        "half",
        "int8_t",
        "int8_t",
        "int8_t",
        "int32_t",
        "int64_t",
        "int32_t",
        "int32_t",
        "int32_t",
        "int32_t",
        "float",
    )
    assert tuple(p.cpp_type for p in params[:12]) == expected_dtypes
    return (
        name,
        params,
        counts,
        scalar_values,
        dict(
            scope="synthetic single-AIC producer, no consumer/FIFO",
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            packed_wrapper=str(wrapper),
            buffers=[
                dict(logical_name=n, cpp_name=p.name, elements=counts[p.name], cpp_type=p.cpp_type)
                for p, n in zip(params[:12], BUFFERS)
            ],
            scalars=[
                dict(argument=i, cpp_name=params[i].name, meaning=arguments[i], value=v)
                for i, v in scalar_values.items()
            ],
            checked_tensor_views=views,
            expected_output_value=8.0,
            expected_output_elements=12288,
            control_source="explicit plan/groups/descriptors and scalar dimensions derived from packed wrapper",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--testcase", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("sim",), default="sim")
    parser.add_argument("--soc-version")
    parser.add_argument("--aicore-arch", default="dav-c220-cube")
    args = parser.parse_args()
    gen = module(Path(__file__).resolve().parent / "tools/gen_profiling_case.py", "score_profiling_generator")
    name, params, counts, scalars, report = bindings(args.input, gen)
    output = args.output_root / "ptoas" / args.testcase
    output.mkdir(parents=True, exist_ok=False)
    kernel_cpp = gen.emit_kernel_cpp(args.input.read_text(), name, False, params)
    (output / f"{args.testcase}_kernel.cpp").write_text(use_cann92_builtin_types(kernel_cpp))
    (output / "launch.cpp").write_text(use_cann92_builtin_types(gen.emit_launch_cpp(name, params)))
    main_cpp = gen.emit_main_cpp(name, params, counts, set(scalars), max(scalars.values()))
    for i, value in scalars.items():
        marker = f"    {params[i].cpp_type} {params[i].name} = 1;"
        assert main_cpp.count(marker) == 1
        main_cpp = main_cpp.replace(marker, f"    {params[i].cpp_type} {params[i].name} = {value};", 1)
    out = params[11].name
    marker = "    ACL_CHECK(aclrtSynchronizeStream(stream));"
    assert main_cpp.count(marker) == 1
    main_cpp = main_cpp.replace(
        marker,
        marker
        + f"""
    ACL_CHECK(aclrtMemcpy({out}Host, fileSize_{out}, {out}Device, fileSize_{out}, ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t score_i = 0; score_i < elemCount_{out}; ++score_i) {{
        if ({out}Host[score_i] != 8.0f) {{
            std::fprintf(stderr, "SCORE_DIAGNOSTIC_MISMATCH index=%zu value=%g\\n", score_i, {out}Host[score_i]);
            rc = 3;
            goto cleanup;
        }}
    }}
    std::printf("SCORE_DIAGNOSTIC_OUTPUT_OK elements=%zu value=8\\n", elemCount_{out});
    {{
        std::FILE* score_report = std::fopen("score_output_check.txt", "w");
        if (!score_report) {{ rc = 4; goto cleanup; }}
        std::fprintf(score_report, "passed elements=%zu value=8\\n", elemCount_{out});
        std::fclose(score_report);
    }}
""",
        1,
    )
    (output / "main.cpp").write_text(main_cpp)
    (output / "CMakeLists.txt").write_text(
        gen._CMAKE_TEMPLATE.replace("@TESTCASE@", args.testcase).replace("@AICORE_ARCH@", args.aicore_arch)
    )
    report["scaffold_compatibility"] = (
        "CANN 9.2/PTO-ISA common types; removed six old typedefs and duplicate MrgSortExecutedNumList fallback only"
    )
    (output / "input_bindings.json").write_text(json.dumps(report, indent=2) + "\n")
    helper_dir = str(Path(__file__).resolve().parent)
    (output / "golden.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        + f"sys.path.insert(0, {helper_dir!r})\n"
        + "from generate_score_sim_case import write_inputs\nwrite_inputs(Path(__file__).parent)\n"
    )
    print(output)


if __name__ == "__main__":
    main()
