# SPDX-License-Identifier: Apache-2.0
"""只读full_v4产物，核查valid_rows、先判slot边界及同GM写后读的真实指令。

这是对已经生成的固定证据文件的审计，不生成算子、不重编译。所列CPP局部
符号用于指明本次编译证据，正式实现仍只有动态高层源码。硬件结果尚未验证。
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "full_v4"


def main():
    result = json.loads((RUN / "result.json").read_text())
    assert result["full_codegen_passed"]
    kernel = RUN / "kernel/kernels/aiv/compressor_output_probe_incore_0.cpp"
    orchestration = RUN / "kernel/orchestration/compressor_output_probe.cpp"
    text, orch = kernel.read_text(), orchestration.read_text()

    def location(fragment):
        assert fragment in text, fragment
        return {"file": str(kernel.relative_to(HERE)), "line": text[: text.index(fragment)].count("\n") + 1}

    dynamic = re.findall(r"// Extract dynamic dim: (\w+)", text)
    assert len(dynamic) == 8
    assert orch.count("rt_submit_aiv_task(") == 1
    assert all(f"params_t0.add_inout(ext_{name});" in orch for name in ("normed", "debug", "cache"))
    # 本次CPP：v160=min(16,end-first)，两个normed列段的GM形状和Tile均使用该值。
    assert "int64_t v160 = v159 < v39 ? v159 : v39;" in text
    assert "v328.SetValidShape(v160, v35);" in text
    assert "v433.SetValidShape(v160, v35);" in text
    assert "v334, v160, v35)" in text and "v439, v160, v35)" in text
    first_store = text.index("TSTORE(v339, v328);")
    rope_store = text.index("TSTORE(v444, v433);")
    barrier = text.index("wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);")
    readback = text.index("TLOAD(v457, v465);")
    assert first_store < rope_store < barrier < readback
    # 两段写回与后续FP32整行读取均由normed原GM的v117描述符取地址；不是DCE空返回。
    assert "__gm__ float* v331 = PTOAS__GLOBAL_TENSOR_DATA(v117);" in text
    assert "__gm__ float* v436 = PTOAS__GLOBAL_TENSOR_DATA(v117);" in text
    assert "__gm__ float* v460 = PTOAS__GLOBAL_TENSOR_DATA(v117);" in text
    assert "v117 = GlobalTensor<float" in text and "(v11, v115, v116);" in text
    # scalar slot读取在compact双界内；page与offset双界在两份副作用之前。
    compact_guard = text.index("if ((v450 >= v38 & v450 < v20) & v450 < v17)")
    slot_page = text.index("int32_t v452 = (v12)[v451];")
    slot_offset = text.index("int32_t v454 = (v12)[")
    page_guard = text.index("if ((v453 >= v38 & v453 < v15 / v53 & v455 >= v38) & v455 < v53)")
    row_guard = text.index("if (v456 < v15)")
    debug_store = text.index("TSTORE(v471, v457);")
    cache_store = text.index("TSTORE(v482, v472);")
    assert compact_guard < slot_page < slot_offset < page_guard < row_guard < readback < debug_store < cache_store
    assert "normed__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor*>(args[10]);" in text
    assert "slots__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor*>(args[11]);" in text
    report = {
        "passed": True,
        "scope": "full_v4实际CPP，只读审计；未重编译、未上卡",
        "dynamic_axes": dynamic,
        "single_AIV_submission": True,
        "inout_outputs": ["normed", "debug", "cache"],
        "valid_rows": location("int64_t v160 = v159 < v39 ? v159 : v39;"),
        "normed_nope_actual_store": location("TSTORE(v339, v328);"),
        "normed_rope_actual_store": location("TSTORE(v444, v433);"),
        "MTE3_to_MTE2_before_readback": location("wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);"),
        "same_normed_GM_readback": location("TLOAD(v457, v465);"),
        "compact_guard_before_slot": location("if ((v450 >= v38 & v450 < v20) & v450 < v17)"),
        "page_offset_guard": location("if ((v453 >= v38 & v453 < v15 / v53 & v455 >= v38) & v455 < v53)"),
        "debug_store": location("TSTORE(v471, v457);"),
        "cache_store": location("TSTORE(v482, v472);"),
        "notes": [
            "cos/sin合同共用一个动态C轴，因此后端合并二者容量检查；Native绑定须保证两者同C。slots保留独立C轴。",
            "normed只写active请求rows；debug/cache只写合法关闭组及合法槽。",
            "规范中的phase跨核可见性由外层处理；本次只证明本核GM读写及编译器pipeline依赖保留。",
        ],
        "device_execution": False,
        "device_precision_or_performance_claim": False,
    }
    (RUN / "compiled_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "dynamic_axes": len(dynamic), "actual_GM_read_after_write": True}))


if __name__ == "__main__":
    main()
