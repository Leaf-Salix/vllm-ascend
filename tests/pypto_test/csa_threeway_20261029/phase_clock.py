# SPDX-License-Identifier: Apache-2.0
"""经用户授权，仅独立诊断副本暴露信号槽并插入既有阶段时钟。"""

import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import regex as re


def artifact_receipt(artifact, codegen):
    """把CPU审计绑定到实际缓存源码和已加载二进制身份，设备运行前重复核验。"""
    from csa_threeway_20261029.clock_receivers import verify_host
    from pypto.backend.pto_backend import _preprocess_ptoas_output

    audit = json.loads((codegen / "audit.json").read_text())
    assert audit["non_clock_source_equal"] and audit["backoff_ticks"] == 150
    assert audit["host_abi"] == verify_host((codegen / "host.cpp").read_text())
    directory = artifact.output_dir
    assert verify_host((directory / "orchestration/decode_csa_spmd.cpp").read_text()) == audit["host_abi"]
    modified = _preprocess_ptoas_output((codegen / "modified.cpp").read_text())
    files = {}
    for role, subdir in (("cube_schedule", "aic"), ("vector_schedule", "aiv")):
        path = directory / f"kernels/{subdir}/{role}.cpp"
        content = path.read_text()
        assert modified in content, "loaded cache did not contain audited instrumentation"
        files[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(artifact=artifact.identity().hex(), source_hashes=files, host_abi=audit["host_abi"])


def instrument_native_wait(source, tools):
    """保留PTOAS原生150-tick退避，只包裹调用取时。"""
    original = source
    changes = []
    pattern = re.compile(r"^(\s*)PTOAS__WAIT_GE_BACKOFF<(\d+)>\(([^\n]+)\);$", re.M)

    def wait(match):
        assert int(match[2]) == 150, "unexpected production backoff"
        replacement = f"{match[1]}CSA_PROFILE_WAIT_BACKOFF<{match[2]}>({match[3]}, csa_wait_ticks, csa_wait_count);"
        changes.append((match[0], replacement))
        return replacement

    source, waits = pattern.subn(wait, source)
    assert waits > 0
    pattern = re.compile(r"^(\s*)pto::comm::TNOTIFY\(([^\n]+)\);$", re.M)

    def notify(match):
        replacement = f"{match[1]}CSA_PROFILE_NOTIFY({match[2]}, csa_wait_ticks, csa_wait_count);"
        changes.append((match[0], replacement))
        return replacement

    source, notifications = pattern.subn(notify, source)
    assert notifications > 0
    declarations = "  uint64_t csa_wait_ticks = 0;\n  uint32_t csa_wait_count = 0;\n"
    for guard in ("  #if defined(__DAV_CUBE__)\n", "  #if defined(__DAV_VEC__)\n"):
        assert source.count(guard) == 1
        source = source.replace(guard, guard + declarations)
    old = tools.CLOCK_HELPERS
    notify_start = old.rfind("template <typename Signal>", 0, old.index("CSA_PROFILE_NOTIFY"))
    helper = r"""
template <uint64_t BackoffTicks, typename Signal>
static AICORE inline void CSA_PROFILE_WAIT_BACKOFF(Signal& signal, int32_t expected,
    uint64_t& ticks, uint32_t& count) {
  const uint64_t start = get_sys_cnt();
  PTOAS__WAIT_GE_BACKOFF<BackoffTicks>(signal, expected);
  ticks += get_sys_cnt() - start;
  ++count;
}
""" + old[notify_start:]
    marker = "AICORE void cube_schedule("
    assert source.count(marker) == 1
    source = source.replace(marker, helper + "\n" + marker)
    reverted = source.replace(helper + "\n" + marker, marker).replace(declarations, "")
    for before, after in reversed(changes):
        reverted = reverted.replace(after, before)
    assert reverted == original, "non-clock source changed"
    return source, dict(waits=waits, notifications=notifications, backoff_ticks=150, non_clock_source_equal=True)


def prepare(source, target):
    assert not target.exists()
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "entry.py"
    tree = ast.parse(path.read_text())
    before = copy.deepcopy(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_spmd")
    index = next(i for i, a in enumerate(fn.args.args) if ast.unparse(a.annotation) == "pl.constexpr")
    annotation = ast.parse("pl.InOut[pl.Tensor[[PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS], pl.INT32]]", mode="eval").body
    fn.args.args.insert(index, ast.arg(arg="clock_signal", annotation=annotation))
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "signal")
    original_value = copy.deepcopy(assignment.value)
    assert ast.unparse(assignment.value).startswith("pl.create_tensor(")
    assignment.value = ast.Name(id="clock_signal", ctx=ast.Load())
    returned = next(n for n in fn.body if isinstance(n, ast.Return))
    returned.value.elts.append(ast.Name(id="clock_signal", ctx=ast.Load()))
    # 撤销三个明确诊断增量后，整个entry AST必须与冻结源相同。
    reverted = copy.deepcopy(tree)
    check = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    check.args.args.pop(index)
    next(
        n for n in check.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "signal"
    ).value = original_value
    next(n for n in check.body if isinstance(n, ast.Return)).value.elts.pop()
    assert ast.dump(reverted) == ast.dump(before)
    ast.fix_missing_locations(tree)
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")


def load_tools(root):
    path = (
        root / "src/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/compile_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("paired_phase_clock_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_hc_detail(target, arm):
    """仅诊断G6输出组流水：复用无消费者的40/41槽，按core区分HC内部取时。"""
    assert arm in ("ours", "nalinaly")
    edits = {}
    cube = target / "cube_schedule.py"
    line = (
        "    partials = hc_linear(x_fp32, hc_packed_input, hc_packed_weight, partials, worker)\n"
        if arm == "ours"
        else "    partials = hc_project_range(x_fp32, hc_fn, partials, worker, GROUPS)\n"
    )
    edits[cube] = [
        (
            line,
            "    signal = publish_phase(signal, PHASE_OA, worker, CORES, generation)\n"
            + line
            + "    signal = publish_phase(signal, PHASE_O_QUANT, worker, CORES, generation)\n",
        )
    ]
    vector = target / "vector_schedule.py"
    line = "    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)\n"
    edits[vector] = [(line, line + "    signal = publish_phase(signal, PHASE_OA, core, CORES, generation)\n")]
    if arm == "ours":
        line = "    hc_packed_input = hc_pack_input_from_bf16(x, hc_packed_input, worker)\n"
        edits[vector].append(
            (line, line + "    signal = publish_phase(signal, PHASE_O_QUANT, core, CORES, generation)\n")
        )
    else:
        before, after = edits[vector][0]
        edits[vector][0] = (
            before,
            after + "    signal = publish_phase(signal, PHASE_O_QUANT, core, CORES, generation)\n",
        )
    for path, changes in edits.items():
        original = path.read_text()
        content = original
        for before, after in changes:
            assert content.count(before) == 1
            content = content.replace(before, after)
        reverted = content
        for before, after in reversed(changes):
            assert reverted.count(after) == 1
            reverted = reverted.replace(after, before)
        assert reverted == original, "non-clock HC source changed"
        path.write_text(content)


@contextlib.contextmanager
def instrument(root, output, package=None):
    from pypto.backend import pto_backend

    tool = load_tools(root)
    original = pto_backend._compile_pto_module
    output.mkdir(exist_ok=False)
    generated = []

    def hooked(code, unit, directory, *args, **kwargs):
        source = original(code, unit, directory, *args, **kwargs)
        if unit != "dynamic_csa_full":
            return source
        (output / "original.cpp").write_text(source)
        from csa_threeway_20261029.clock_receivers import verify

        assert package is not None
        receivers = verify(source)
        changed, audit = instrument_native_wait(source, tool)
        audit["receivers"] = receivers
        (output / "modified.cpp").write_text(changed)
        (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        generated.append(Path(directory) / "orchestration/decode_csa_spmd.cpp")
        return changed

    pto_backend._compile_pto_module = hooked
    try:
        yield
        from csa_threeway_20261029.clock_receivers import verify_host

        for path in generated:
            host = path.read_text()
            audit = json.loads((output / "audit.json").read_text())
            audit["host_abi"] = verify_host(host)
            (output / "host.cpp").write_text(host)
            (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    finally:
        pto_backend._compile_pto_module = original
