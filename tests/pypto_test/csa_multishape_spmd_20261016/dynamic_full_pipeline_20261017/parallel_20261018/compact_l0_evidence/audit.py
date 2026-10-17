"""比较直接源码、实际 helper 的 operand 序列及现有物理分配；无设备执行。"""

import ast
import copy
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace


E = Path(__file__).resolve().parent
W = E.parents[2]
P = W / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
D = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017")
BASE = D / "parallel_20261018/baseline_source"
BPTO = D / "parallel_20261018/current_clock_h131072_b16_v1/build_output/decode_csa_spmd_jqlk5eey/ptoas/dynamic_csa_full.pto"
CPTO = E / "cpu_long_b16_v1/kernel/ptoas/dynamic_csa_full.pto"
MODULE = "compressor_projection_stages.py"
HELPER = "indexer_compressor_project_stage"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


@dataclass
class Tile:
    shape: tuple
    valid: tuple
    source: tuple | None = None
    acc_id: int | None = None


class Tensor:
    def __init__(self, owner, name, rows, cols):
        self.owner, self.name, self.shape = owner, name, (rows, cols)

    def __getitem__(self, indices):
        offsets = [index.start for index in indices]
        shape = [index.stop - index.start for index in indices]
        return self.owner.load(self, offsets, shape)

    def __setitem__(self, indices, value):
        assert tuple(index.stop - index.start for index in indices) == value.shape
        self.owner.store(value, [index.start for index in indices], self)


class PL:
    """只执行实际 helper 的标量控制流，记录访问与 MMA；不模拟浮点舍入。"""

    def __init__(self):
        self.FP32 = "FP32"
        self.MemorySpace = SimpleNamespace(Mat="Mat", Left="Left", Right="Right", Acc="Acc")
        self.tensor = SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis])
        self.tile = SimpleNamespace(move=lambda tile, **kw: tile,
                                    transpose_view=lambda tile: replace(tile, shape=tile.shape[::-1], valid=tile.valid[::-1]),
                                    matmul_acc=self.matmul_acc)
        self.trace, self.loads, self.stores = [], [], []
        self.acc_count = 0

    range = staticmethod(range)
    min = staticmethod(min)

    @staticmethod
    def pipeline(*args, **kwargs):
        return range(*args)

    def create_tile(self, shape, **kwargs):
        tile = Tile(tuple(shape), tuple(shape), acc_id=self.acc_count)
        self.acc_count += 1
        return tile

    create_tensor = create_tile

    def load(self, tensor, offsets, shape, valid_shape=None, **kwargs):
        valid = tuple(valid_shape or shape)
        for offset, size, extent in zip(offsets, valid, tensor.shape):
            assert 0 <= offset and 0 < size and offset + size <= extent
        self.loads.append((tensor.name, tuple(offsets), tuple(shape), valid))
        return Tile(tuple(shape), valid, (tensor.name, tuple(offsets)))

    def slice(self, tensor, shape, offsets, **kwargs):
        return self.load(tensor, offsets, shape, **kwargs)

    def matmul_acc(self, acc, left, right, b_trans=False, init_cond=False):
        right_shape = right.shape[::-1] if b_trans else right.shape
        assert left.shape == (16, 512) and right_shape == (512, 32)
        assert acc.shape == (16, 32) and acc.valid == (16, 32)
        assert left.valid[0] in range(1, 17)
        self.trace.append((acc.acc_id, left.source, right.source, left.shape, left.valid,
                           right_shape, bool(init_cond)))
        return acc

    def store(self, tile, offsets, tensor):
        assert tile.shape == (16, 32)
        assert offsets[0] + 16 <= tensor.shape[0] and offsets[1] + 32 <= tensor.shape[1]
        self.stores.append((tensor.name, tuple(offsets), tile.shape, tile.acc_id))


def executable(path):
    node = copy.deepcopy(function(path, HELPER))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    code = compile(module, str(path), "exec")
    return code


def trace(code, tokens, workers):
    all_traces = []
    for worker in range(workers):
        pl = PL()
        env = {"pl": pl, "D": 4096, "K_TILE": 512, "INDEXER_M_TILE": 16,
               "INDEXER_N_TILE": 32, "INDEXER_OUT_DIM": 256}
        exec(code, env)
        args = [Tensor(pl, "x", tokens, 4096), Tensor(pl, "wkv", 256, 4096),
                Tensor(pl, "wgate", 256, 4096), Tensor(pl, "kv_out", (tokens + 15) // 16 * 16, 256),
                Tensor(pl, "gate_out", (tokens + 15) // 16 * 16, 256)]
        env[HELPER](*args, worker, workers)
        all_traces.append((pl.trace, pl.loads, pl.stores))
    return all_traces


def physical(path, package):
    functions = {}
    for file in package.glob("*.py"):
        functions[file.name] = [(node.lineno, node.end_lineno, node.name) for node in ast.parse(file.read_text()).body
                                if isinstance(node, ast.FunctionDef)]
    records = []
    fn = ""
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        f = re.search(r"func.func @(\w+)", line)
        if f:
            fn = f[1]
        if "pto.alloc_tile" not in line:
            continue
        m = re.search(r"addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        loc = re.search(r' loc\("([^\"]+)":(\d+):\d+\)', line)
        assert m and loc, (path, line_number)
        name, source_line = Path(loc[1]).name, int(loc[2])
        source_function = next((f for lo, hi, f in functions.get(name, []) if lo <= source_line <= hi), "outside_helper")
        bits = {"bf16": 16, "f16": 16, "f32": 32, "i8": 8, "i16": 16, "i32": 32,
                "ui32": 32, "i64": 64, "i1": 1}[m[3]]
        address, rows, cols = int(m[1]), int(m[4]), int(m[5])
        records.append({"function": fn, "source_file": name, "source_function": source_function,
                        "space": m[2], "dtype": m[3], "shape": [rows, cols], "address": address,
                        "end": address + rows * cols * bits // 8, "ir_line": line_number, "source_line": source_line})
    return records


def summary(records):
    spans = defaultdict(int)
    for row in records:
        key = row["function"] + "/" + row["space"]
        spans[key] = max(spans[key], row["end"])
    return dict(spans)


def allocation_counter(records):
    return Counter((r["function"], r["source_file"], r["source_function"], r["space"], r["dtype"],
                    tuple(r["shape"]), r["address"]) for r in records)


def main():
    source_rows = [{"file": path.name, "equal": path.read_bytes() == (BASE / path.name).read_bytes(),
                    "candidate_bytes": path.stat().st_size, "baseline_bytes": (BASE / path.name).stat().st_size}
                   for path in sorted(P.glob("*.py"))]
    assert [r["file"] for r in source_rows if not r["equal"]] == [MODULE]
    assert ast.dump(function(P / MODULE, "compressor_project_stage")) == ast.dump(function(BASE / MODULE, "compressor_project_stage"))
    assert ast.dump(function(P / MODULE, HELPER).args) == ast.dump(function(BASE / MODULE, HELPER).args)
    cases = []
    old, new = executable(BASE / MODULE), executable(P / MODULE)
    token_cases = list(range(34)) + [63, 64, 65, 95, 96, 97, 127, 128, 129, 143, 144, 145, 167, 168, 169, 191, 192, 193]
    for tokens in token_cases:
        for workers in (1, 16, 24):
            a, b = trace(old, tokens, workers), trace(new, tokens, workers)
            assert a == b, (tokens, workers)
            cases.append({"tokens": tokens, "workers": workers, "mma_calls": sum(len(x[0]) for x in a), "equal": True})
    baseline, candidate = physical(BPTO, BASE), physical(CPTO, P)
    ca, cb = allocation_counter(baseline), allocation_counter(candidate)
    removed = list((ca - cb).elements())
    added = list((cb - ca).elements())
    assert all(r[2] == HELPER for r in removed + added)
    assert summary(baseline) == summary(candidate)
    selected = [r for r in candidate if r["source_function"] == HELPER and r["space"] == "right"]
    assert [r["address"] for r in selected] == [0, 32768, 0, 0]
    cpp = E / "cpu_long_b16_v1/kernel/kernels/aic/cube_schedule.cpp"
    lines = cpp.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "// pto: %left0_" in line)
    end = next(i for i, line in enumerate(lines) if "// pto: %right_gate1_" in line) + 34
    excerpt = "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end)) + "\n"
    (E / "candidate_cpp_excerpt.txt").write_text(excerpt)
    report = {
        "passed": True, "source_package": str(P), "baseline_source": str(BASE), "sources": source_rows,
        "only_indexer_helper_changed": True, "helper_signature_unchanged": True,
        "trace_method": "抽取真实 baseline/candidate helper AST 执行标量控制流，分别比较 GM load、MMA operand/shape/valid/init、store 三份列表及各自内部顺序；不比较三类操作合并后的完整指令时序，搬运与 MMA 的相对时刻有意改变。不计算浮点值，不代替设备精度。",
        "trace_cases": cases, "case_count": len(cases),
        "physical_baseline": str(BPTO), "physical_candidate": str(CPTO),
        "physical_comparison": "按 AIC/AIV、源helper、空间、dtype、物理形状、地址的多重集直接对照；排除变量名与源行号重命名。",
        "all_other_helper_allocations_equal": True,
        "removed_allocations": removed, "added_allocations": added,
        "baseline_address_high_water": summary(baseline), "candidate_address_high_water": summary(candidate),
        "high_water_limit": "addr + 物理行列×dtype字节的最大上界；不是动态活跃峰值、占用率或性能值。完整 compiler 资源合法性另已通过。",
        "candidate_right_allocations": selected,
        "cpp": str(cpp), "cpp_excerpt": str(E / "candidate_cpp_excerpt.txt"),
        "cpp_manual_review": "KV1 TMOV 在8036先于首MMA8042/8044；gate0 MMA8068/8070后KV1 MMA8074直接消费已驻留Right，原gate0→KV1搬运覆盖等待消失。gate1 Right仍在8089等gate0消费后复用0；MMA8073前有PIPE_M barrier，未将其称为可删除等待。",
        "device_execution": False, "device_precision_validated": False, "performance_measured": False,
    }
    (E / "audit_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "trace_cases": len(cases), "source_files": len(source_rows),
                      "removed_allocations": removed, "added_allocations": added,
                      "high_water": summary(candidate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
