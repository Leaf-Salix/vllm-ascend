# SPDX-License-Identifier: Apache-2.0
"""测试用 GM 信号读侧：比较 DCCI 后读取与 A3 ld_dev，支持未就绪退避。

保留原 TNOTIFY 的 DCCI+DSB 发布及调用者对 payload 的缓存维护。这里只
替换单元素 GE 等待，不修改算术、不使用 st_dev 写信号、不声称缓存一致性。
绕过缓存的依据是 A3 PTO-ISA common/syncall_soft.hpp 的 ReadCounter；
cached 路径每次读取前仍执行 DCCI，不假定 Scalar 缓存具有跨核一致性。
"""

import contextlib
import json
from pathlib import Path


HELPER = r'''
template <typename Signal>
static AICORE inline void CSA_WAIT_LOCAL_GM(Signal& signal, int32_t expected, pto::comm::WaitCmp cmp) {
  // 单写者单调进度。写侧仍由 TNOTIFY 清理/失效并 DSB，payload 由调用方维护。
  PTO_ASSERT(cmp == pto::comm::WaitCmp::GE, "CSA local wait only supports GE");
  auto ptr = reinterpret_cast<__gm__ uint32_t*>(signal.data());
  uint32_t spin = 0;
  while (static_cast<int32_t>(ld_dev(ptr, 0)) < expected) {
    PTO_ASSERT(spin < 100000000, "CSA local wait spin limit");
    if ((++spin & 63) == 0) {
      pipe_barrier(PIPE_ALL);
    }
  }
}
'''


def audit_wait_artifact(artifact_dir, change_dir, backoff_ticks, *, cached, ramp_backoff=False):
    """执行前核对真正交给CCEC的AIC/AIV源码，不能仅相信命令行选项。

    ptoas目录保存的是替换前源码；实际等待策略在kernels下。提前解析JIT
    artifact会把未替换的版本缓存起来，因此同时检查编译回调记录和两份源码。
    """
    change_path = Path(change_dir) / "change.json"
    if not change_path.is_file():
        raise ValueError("等待策略编译回调没有执行，禁止使用默认等待版本测量")
    change = json.loads(change_path.read_text())
    expected = {"reader": "dcci_then_volatile_load" if cached else "ld_dev",
                "backoff_syscnt_ticks": backoff_ticks, "ramp_backoff": ramp_backoff}
    if any(change.get(key) != value for key, value in expected.items()):
        raise ValueError("等待策略编译记录与请求不符")
    kernels = {}
    for kind in ("aic", "aiv"):
        path = Path(artifact_dir) / "kernels" / kind / f"csa_static_{kind}.cpp"
        source = path.read_text()
        calls = source.count("CSA_WAIT_LOCAL_GM(") - 1  # 去掉helper定义。
        if calls != change["wait_sites"] or calls <= 0 or "pto::comm::TWAIT(" in source:
            raise ValueError(f"{kind}实际源码没有完整应用等待策略")
        if cached:
            if ("if (*ptr >= expected) break;" not in source
                    or "dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);" not in source):
                raise ValueError(f"{kind}实际源码缺少DCCI后读取")
        elif "ld_dev(ptr, 0)" not in source:
            raise ValueError(f"{kind}实际源码缺少ld_dev读取")
        if backoff_ticks:
            delay = "csa_backoff" if ramp_backoff else str(backoff_ticks)
            if f"get_sys_cnt() - csa_wait_start) < {delay})" not in source:
                raise ValueError(f"{kind}实际源码的轮询退避不符")
        elif "csa_wait_start" in source:
            raise ValueError(f"{kind}实际源码出现未请求的轮询退避")
        kernels[kind] = {"source": str(path), "wait_sites": calls}
    return {"passed": True, **expected, "kernels": kernels,
            "checked_before_first_launch": True}


@contextlib.contextmanager
def bypass_signal_wait(output, backoff_ticks=0, *, cached=False, ramp_backoff=False):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if not 0 <= backoff_ticks <= 1000:
        raise ValueError("诊断轮询间隔必须在 0..1000 个 SYS_CNT tick 内")
    if ramp_backoff and (not cached or backoff_ticks < 4):
        raise ValueError("递增退避只用于DCCI读取，最大间隔至少4tick")

    def compile_with_wait(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "csa_static_group":
            return source
        if cached and backoff_ticks == 100 and not ramp_backoff:
            from production_sources import load_operator_module

            # 历史静态复现明确请求100；不能跟随正式动态默认间隔的后续更新。
            transformed, change = load_operator_module("wait_codegen").rewrite_wait_source(
                source, backoff_ticks=backoff_ticks
            )
            (directory / "original.cpp").write_text(source)
            (directory / "bypass_wait.cpp").write_text(transformed)
            (directory / "change.json").write_text(
                json.dumps(change, ensure_ascii=False, indent=2) + "\n")
            return transformed
        needle = "pto::comm::TWAIT("
        count = source.count(needle)
        if not count:
            raise ValueError("未找到融合任务的等待调用")
        (directory / "original.cpp").write_text(source)
        source = source.replace(needle, "CSA_WAIT_LOCAL_GM(")
        marker = "using namespace pto;"
        if source.count(marker) != 1:
            raise ValueError("无法唯一放置测试 helper")
        helper = HELPER
        if cached:
            # 保留 PTO-ISA TWAIT 的 DCCI+volatile GM 读，只在尚未就绪时
            # 给轮询增加间隔。不会把普通 cached load 当成具有跨核一致性。
            helper = helper.replace("reinterpret_cast<__gm__ uint32_t*>",
                                    "reinterpret_cast<volatile __gm__ int32_t*>")
            helper = helper.replace("  while (static_cast<int32_t>(ld_dev(ptr, 0)) < expected) {", '''  while (true) {
    __asm__ __volatile__("" ::: "memory");
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    if (*ptr >= expected) break;''')
        if backoff_ticks:
            delay = str(backoff_ticks)
            advance = ""
            if ramp_backoff:
                # 每个新依赖先快速复查，持续未就绪才增至原最大间隔。
                # 4/8/16/32/64/100tick；读前DCCI、写侧发布和payload维护不变。
                helper = helper.replace("  uint32_t spin = 0;", "  uint32_t spin = 0;\n  uint32_t csa_backoff = 4;")
                delay = "csa_backoff"
                advance = (f"    csa_backoff = csa_backoff * 2 < {backoff_ticks} "
                           f"? csa_backoff * 2 : {backoff_ticks};\n")
            helper = helper.replace("    PTO_ASSERT(spin", (
                "    // 首次读取不延迟；只有尚未就绪时，短暂只读本核计时寄存器。\n"
                "    const uint64_t csa_wait_start = get_sys_cnt();\n"
                f"    while ((uint64_t)(get_sys_cnt() - csa_wait_start) < {delay}) {{\n"
                "      __asm__ __volatile__(\"\" ::: \"memory\");\n"
                "    }\n"
                + advance +
                "    PTO_ASSERT(spin"))
        source = source.replace(marker, marker + "\n" + helper)
        (directory / "bypass_wait.cpp").write_text(source)
        (directory / "change.json").write_text(json.dumps(
            {"wait_sites": count, "reader": "dcci_then_volatile_load" if cached else "ld_dev",
             "backoff_syscnt_ticks": backoff_ticks,
             "ramp_backoff": ramp_backoff,
             "initial_backoff_syscnt_ticks": 4 if ramp_backoff else backoff_ticks,
             "writer_unchanged": True,
             "payload_maintenance_unchanged": True, "device_validated": False},
            ensure_ascii=False, indent=2) + "\n")
        return source

    pto_backend._compile_pto_module = compile_with_wait
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original
