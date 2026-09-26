"""Standalone A3 TDIV precision observation; no model, weights or vLLM imports."""

import argparse
import json
import os
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import pypto.language as pl

ROWS = 16
COLS = 256


@pl.jit(auto_scope=False)
def vector_division(
    numerator: pl.Tensor[[ROWS, COLS], pl.FP32],
    denominator: pl.Tensor[[ROWS, COLS], pl.FP32],
    normal: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    precise: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for row in pl.spmd(ROWS):
        left = numerator[row : row + 1, :]
        right = denominator[row : row + 1, :]
        normal[row : row + 1, :] = pl.div(left, right)
        precise[row : row + 1, :] = pl.div(left, right, high_precision=True)
    return normal, precise


@pl.jit(auto_scope=False)
def scalar_division(
    numerator: pl.Tensor[[ROWS, COLS], pl.FP32],
    denominator: pl.Tensor[[ROWS, COLS], pl.FP32],
    result: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for row in pl.spmd(ROWS):
        values = pl.full([1, COLS], dtype=pl.FP32, value=0.0)
        for column in pl.range(COLS):
            left = pl.read(numerator, [row, column])
            right = pl.read(denominator, [row, column])
            pl.write(values, [0, column], left / right)
        result[row : row + 1, :] = values
    return result


def git_state(path):
    if path is None:
        return None
    result = {"path": str(path)}
    for name, arguments in (("commit", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"])):
        process = subprocess.run(["git", "-C", str(path), *arguments], capture_output=True, text=True)
        result[name] = process.stdout.strip() if process.returncode == 0 else None
    return result


def exact_reference(torch, numerator, denominator):
    """Check round-to-nearest-even against exact rational quotients."""
    reference = (numerator.double() / denominator.double()).float()
    for left, right, bits in zip(
        numerator.flatten().tolist(), denominator.flatten().tolist(), reference.view(torch.int32).flatten().tolist()
    ):
        quotient = Fraction.from_float(left) / Fraction.from_float(right)
        candidates = (bits - 1, bits, bits + 1)
        chosen = min(
            candidates,
            key=lambda candidate: (
                abs(quotient - Fraction.from_float(struct.unpack("<f", struct.pack("<I", candidate))[0])),
                candidate & 1,
            ),
        )
        if chosen != bits:
            raise AssertionError("FP64-to-FP32 candidate disagrees with exact rational rounding")
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isa-root", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    os.chdir(out)  # Keep generated PTO IR and C++ beside the numerical report.

    import pypto
    import pypto.torch
    import torch
    import torch_npu
    from pypto.runtime import RunConfig

    generator = torch.Generator().manual_seed(20260922)
    denominator_bits = torch.randint(111, 144, (ROWS, COLS), generator=generator, dtype=torch.int32) * (
        1 << 23
    ) + torch.randint(0, 1 << 23, (ROWS, COLS), generator=generator, dtype=torch.int32)
    denominator = denominator_bits.view(torch.float32)
    numerator = torch.ones_like(denominator)
    numerator[8:12] = 127.0
    numerator[12:] = (
        torch.randint(111, 144, (4, COLS), generator=generator, dtype=torch.int32) * (1 << 23)
        + torch.randint(0, 1 << 23, (4, COLS), generator=generator, dtype=torch.int32)
    ).view(torch.float32)
    reference = exact_reference(torch, numerator, denominator)
    for kernel in (vector_division, scalar_division):
        (out / f"{kernel.__name__}_lowered.py").write_text(str(kernel.lower(config=RunConfig(platform="a2a3"))))

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    with torch.inference_mode():
        left, right = numerator.to(device), denominator.to(device)
        normal, precise, scalar = (torch.empty_like(left) for _ in range(3))
        vector_division(left, right, normal, precise)
        scalar_division(left, right, scalar)
        torch.npu.synchronize()
        normal, precise, scalar = (value.cpu() for value in (normal, precise, scalar))

    def difference(actual, expected):
        delta = (actual.view(torch.int32).long() - expected.view(torch.int32).long()).abs()
        return {"different_elements": int((delta != 0).sum()), "max_ulp": int(delta.max())}

    comparisons = {
        "high_vs_default": difference(precise, normal),
        "default_vs_exact_fp32": difference(normal, reference),
        "high_vs_exact_fp32": difference(precise, reference),
        "scalar_vs_exact_fp32": difference(scalar, reference),
    }
    changed = (precise != reference).nonzero()[:12].tolist()
    examples = []
    for row, column in changed:
        entry = {"row": row, "column": column}
        for name, tensor in (
            ("numerator", numerator),
            ("denominator", denominator),
            ("default", normal),
            ("high", precise),
            ("scalar", scalar),
            ("reference", reference),
        ):
            entry[name] = float(tensor[row, column])
            entry[f"{name}_bits"] = f"0x{int(tensor.view(torch.int32)[row, column]):08x}"
        examples.append(entry)

    evidence = {}
    for path in sorted(out.glob("build_output/**/ptoas/*")):
        if path.suffix not in (".pto", ".cpp"):
            continue
        lines = [
            line.strip() for line in path.read_text().splitlines() if "tdiv" in line.lower() or "division_mode" in line
        ]
        evidence[str(path.relative_to(out))] = {
            "size_bytes": path.stat().st_size,
            "division_lines": lines,
        }
    isa = git_state(args.isa_root)
    if args.isa_root:
        for relative in ("include/pto/npu/a2a3/TDiv.hpp", "docs/isa/TDIV.md"):
            path = args.isa_root / relative
            if path.exists():
                target = out / ("isa_" + path.name)
                shutil.copyfile(path, target)
                isa[relative] = {"evidence": target.name, "size_bytes": path.stat().st_size}
    ptoas = shutil.which("ptoas")
    report = {
        "status": "REPRODUCED"
        if comparisons["high_vs_default"]["different_elements"] == 0
        and comparisons["high_vs_exact_fp32"]["different_elements"] > 0
        else "NOT_REPRODUCED",
        "scope": "A3 FP32 tensor/tensor TDIV; not a full CSA accuracy or performance test",
        "source": git_state(source.parent),
        "shape": [ROWS, COLS],
        "seed": 20260922,
        "exact_rational_checked": ROWS * COLS,
        "inputs": "positive finite normal FP32, nonzero denominators, normal finite quotients",
        "device": args.device,
        "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "pypto": git_state(Path(pypto.__file__).parent),
        "pto_isa": isa,
        "ptoas": {"path": ptoas, "version": subprocess.check_output([ptoas, "--version"], text=True).strip()},
        "cann_path": os.environ.get("ASCEND_HOME_PATH"),
        "comparisons": comparisons,
        "examples": examples,
        "generated_code": evidence,
    }
    torch.save(
        {
            "numerator": numerator,
            "denominator": denominator,
            "default": normal,
            "high": precise,
            "scalar": scalar,
            "reference": reference,
        },
        out / "tensors.pt",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
