# SPDX-License-Identifier: Apache-2.0
"""复用正式测试的阶段时钟，补充原位 FFTS 指令发射耗时。

只改本次编译返回的 C++；不增加 fence，不改算法、流水事件或正式源码。
wait_flag_dev 前后时钟的差命名为 issue，不承诺它包含后端流水的全部阻塞。
槽 16..19 保留原 hook，20/21 记录累计 FFTS issue ticks/count；仍只写第二行。
"""

import contextlib
import json
import re


def instrument_ffts(source):
    """在已插阶段时钟的 C++ 上增加计数，所有替换必须命中已知结构。"""
    counter = "  uint32_t csa_wait_count = 0;\n"
    assert source.count(counter) == 2, "必须分别定位 AIC/AIV 的计数器"
    source = source.replace(
        counter,
        counter + "  uint64_t csa_ffts_issue_ticks = 0;\n  uint32_t csa_ffts_issue_count = 0;\n",
    )
    signature = "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count) {"
    assert source.count(signature) == 1
    source = source.replace(
        signature,
        "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count,\n"
        "    uint64_t ffts_issue_ticks, uint32_t ffts_issue_count) {",
    )
    stores = "  ptr[19] = count;\n"
    assert source.count(stores) == 1
    source = source.replace(
        stores,
        stores + "  ptr[20] = static_cast<uint32_t>(ffts_issue_ticks);\n  ptr[21] = ffts_issue_count;\n",
    )
    pattern = re.compile(r"^(\s*)CSA_PROFILE_NOTIFY\(([^\n]+), csa_wait_ticks, csa_wait_count\);$", re.M)
    source, notifications = pattern.subn(
        lambda m: f"{m[1]}CSA_PROFILE_NOTIFY({m[2]}, csa_wait_ticks, csa_wait_count, "
        "csa_ffts_issue_ticks, csa_ffts_issue_count);",
        source,
    )
    assert notifications > 0
    pattern = re.compile(r"^([ \t]*)wait_flag_dev\(([^\n]+)\);$", re.M)

    def replace_wait(match):
        indent, argument = match.groups()
        return (
            indent
            + "{ // CSA_FFTS_ISSUE_PROBE: no added fence\n"
            + indent
            + "  const uint64_t csa_ffts_issue_start = get_sys_cnt();\n"
            + indent
            + f"  wait_flag_dev({argument});\n"
            + indent
            + "  csa_ffts_issue_ticks += get_sys_cnt() - csa_ffts_issue_start;\n"
            + indent
            + "  csa_ffts_issue_count += 1;\n"
            + indent
            + "}"
        )

    source, waits = pattern.subn(replace_wait, source)
    assert waits > 0
    return source, dict(
        ffts_wait_sites=waits,
        ffts_notify_sites=notifications,
        extra_signal_columns=[20, 21],
        added_fences=0,
        meaning="wait_flag_dev 原位 issue 时间；不声称等于 FIFO 完整阻塞时间",
    )


@contextlib.contextmanager
def compilation_probe(directory):
    """仅在当前进程/当前编译上下文扩展原 hook，退出后恢复它。"""
    import compile_diagnostics

    original = compile_diagnostics.instrument_clock

    def extended(source, *, wait_backoff):
        source, audit = original(source, wait_backoff=wait_backoff)
        source, audit["ffts_issue"] = instrument_ffts(source)
        return source, audit

    compile_diagnostics.instrument_clock = extended
    try:
        with compile_diagnostics.compilation_probe(directory, wait_backoff=True, phase_clock=True):
            yield
    finally:
        compile_diagnostics.instrument_clock = original


def export_clock(signal, destination, phases):
    """沿用原时钟导出，附加 FFTS 累计/端点差分；不从 other 中扣除。"""
    from compile_diagnostics import export_clock as export_original

    report = export_original(signal, destination, phases)
    path = destination / "phase_clock.json"
    rows = json.loads(path.read_text())
    for row in rows:
        values = signal[row["phase"], row["core"]].tolist()
        row["cumulative_ffts_issue_us"] = (values[20] & 0xFFFFFFFF) / 50
        row["cumulative_ffts_issue_calls"] = values[21] & 0xFFFFFFFF
    for core in range(72):
        steps = sorted((r for r in rows if r["core"] == core), key=lambda r: r["notify_start_us"])
        for before, after in zip(steps, steps[1:]):
            after["ffts_issue_us"] = after["cumulative_ffts_issue_us"] - before["cumulative_ffts_issue_us"]
            after["ffts_issue_calls"] = after["cumulative_ffts_issue_calls"] - before["cumulative_ffts_issue_calls"]
            assert after["ffts_issue_us"] >= 0 and after["ffts_issue_calls"] >= 0
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    report.update(
        ffts_issue_recorded=True,
        other_includes_ffts=True,
        other_is_pure_compute=False,
        limitation="原位 issue 时间不保证覆盖 FIFO 全部阻塞；other 仍包含 FFTS、搬运、运算及插桩开销",
    )
    return report
