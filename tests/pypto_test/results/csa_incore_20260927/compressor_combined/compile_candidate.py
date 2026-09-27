"""Compile an isolated Native-style combined KV/gate projection candidate."""

import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf"
SOURCE = Path("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf")


def prepare(name, width):
    relative = SOURCE / f"{name}.py"
    old = subprocess.check_output(["git", "show", f"da2e2368:{relative}"], cwd=REPO, text=True)
    start = old.index("            kv_acc = pl.create_tensor(")
    end = old.index("\n    return _kv_score_tid", start)
    body = f'''            x_rows = pl.min(MM_B_TILE, bs - global_row0)
            # Peel K0 so the accumulator inherits the runtime row pitch
            # from matmul, including the final partially valid M tile.
            first_x = pl.load(
                x_flat, [global_row0, 0], [MM_B_TILE, K_TILE],
                valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat,
            )
            first_weights = pl.create_tile(
                [2 * {width}, K_TILE], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat
            )
            first_weights = pl.gather_row(first_weights, wkv, [0, 0], [o0, 0], [{width}, K_TILE])
            first_weights = pl.gather_row(first_weights, wgate, [{width}, 0], [o0, 0], [{width}, K_TILE])
            first_weights_t = pl.tile.transpose_view(first_weights)
            combined_acc = pl.matmul(first_x, first_weights_t, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, D // K_TILE, stage=2):
                k0 = kb * K_TILE
                x_tile = pl.load(
                    x_flat, [global_row0, k0], [MM_B_TILE, K_TILE],
                    valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat,
                )
                # Native packs KV and gate columns into one L1 panel and
                # computes both with the same Mmad; GM weights stay separate.
                packed_weights = pl.create_tile(
                    [2 * {width}, K_TILE], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat
                )
                packed_weights = pl.gather_row(
                    packed_weights, wkv, [0, 0], [o0, k0], [{width}, K_TILE]
                )
                packed_weights = pl.gather_row(
                    packed_weights, wgate, [{width}, 0], [o0, k0], [{width}, K_TILE]
                )
                packed_weights_t = pl.tile.transpose_view(packed_weights)
                combined_acc = pl.matmul_acc(combined_acc, x_tile, packed_weights_t)
            kv_piece = pl.tile.slice(
                combined_acc, [MM_B_TILE, {width}], [0, 0], valid_shape=[x_rows, {width}]
            )
            score_piece = pl.tile.slice(
                combined_acc, [MM_B_TILE, {width}], [0, {width}], valid_shape=[x_rows, {width}]
            )
            pl.store(kv_piece, [global_row0, o0], kv_proj_pad)
            pl.store(score_piece, [global_row0, o0], score_proj_pad)
'''
    candidate = old[:start] + body + old[end:]
    target = ROOT / f"{name}.py"
    target.write_text(candidate)
    patch = "".join(difflib.unified_diff(old.splitlines(True), candidate.splitlines(True),
                                         fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    return target, patch


def main():
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    patches = []
    for name, width in (("decode_compressor_ratio4", "OUT_TILE"),
                        ("decode_indexer_compressor", "PROJ_OUT_TILE")):
        target, patch = prepare(name, width)
        patches.append(patch)
        qualified = f"{PACKAGE}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    (ROOT / "candidate.patch").write_text("".join(patches))
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
