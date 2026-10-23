# SPDX-License-Identifier: Apache-2.0
"""比较合并版与已通过设备门禁的两个分档；只容许明确的生成标识符差异。"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABI_FOOTER = (
    "\n#include <type_traits>\n"
    "static_assert(std::is_same_v<decltype(&aicpu_orchestration_entry), void (*)(const ChipTaskArgs&)>);\n"
    'extern "C" __attribute__((visibility("default")))\n'
    'const char* pypto_kernel_abi_descriptor() { return "pypto-kernel-abi-v1:'
)


def canonical(path, *, device_orchestration=False):
    text = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("//"))
    if device_orchestration and ABI_FOOTER in text:
        text, footer = text.split(ABI_FOOTER, 1)
        assert re.fullmatch(r'[0-9a-f]{64}"; \}', footer.strip())
    identifiers = {}

    def rename(match):
        identifier = match.group(0)
        if identifier not in identifiers:
            identifiers[identifier] = f"_inlineCanonical{len(identifiers)}"
        return identifiers[identifier]

    return re.sub(r"_inline\d+", rename, text).rstrip()


def main():
    abi_report = json.loads((HERE / "scheduling_v36_kernel_abi.json").read_text())
    assert abi_report["passed"]
    reports = []
    for case, variant in (("h262144_b16_s6", "v34"), ("h131072_b24_s6", "v32")):
        cpu = HERE / f"runs/cpu_v36_{case}/result.json"
        assert json.loads(cpu.read_text())["passed"]
        new = next((cpu.parent / "build_output").glob("decode_csa_tnd_*"))
        proof = next(c for c in abi_report["cases"] if c["case"] == case)
        old = Path(proof["reference_build"])
        assert str(new) == proof["cpu_build"] and proof["abi_equal"] and proof["lowered_params_equal"]
        meta = json.loads((old / "compiled_meta.json").read_text())
        assert proof["kernel_abi"] == meta["kernel_abi"]
        paths = sorted(f.relative_to(old) for f in (old / "kernels").rglob("*.cpp"))
        assert paths == sorted(f.relative_to(new) for f in (new / "kernels").rglob("*.cpp"))
        paths.append(Path("orchestration/decode_csa_tnd.cpp"))
        rows = []
        for relative in paths:
            before = canonical(old / relative, device_orchestration=relative.parts[0] == "orchestration")
            after = canonical(new / relative)
            extra = None
            if before != after and variant == "v34" and str(relative) == "kernels/aiv/o_quant.cpp":
                # 同一Tensor参数槽的SSA返回别名变化；数学体和ABI必须仍完全相同。
                assert "o_r__rv_v4_tensor = reinterpret_cast<__gm__ Tensor*>(args[0])" in before
                assert "o_r__rv_v9_tensor = reinterpret_cast<__gm__ Tensor*>(args[0])" in after
                before = before.replace("o_r__rv_v4", "o_r__rv_v9")
                extra = "o_quant wrapper: o_r__rv_v4 -> o_r__rv_v9, both Tensor args[0]"
            assert before == after, (case, relative)
            rows.append({"file": str(relative), "equal": True, "extra_wrapper_rename": extra})
        reports.append(
            dict(
                case=case,
                reference_candidate=variant,
                cpu_result=str(cpu),
                reference_build=str(old),
                combined_build=str(new),
                kernel_abi_equal=True,
                compared_files=rows,
            )
        )
        print(case, len(rows), "CPP files and derived kernel ABI match validated implementation")
    report = dict(
        candidate="aicpu_v36_capacity_rectangle",
        passed=True,
        adoption_record="scheduling_v36_adoption.json",
        cases=reports,
        normalization=(
            "Whole-line comments removed. Generated _inlineN site identifiers bijectively renamed in first-occurrence "
            "order. Device-only ABI footer excluded only after real pre-pass IR ABI and lowered params match. "
            "One o_quant wrapper SSA root rename explicitly recorded. No opcode, constant, layout, task order, "
            "launch flag, argument slot, tensor dtype/shape or return alias is ignored."
        ),
        abi_source="scheduling_v36_kernel_abi.json",
        off_domain_evidence="scheduling_v32_closed_domain.json",
        gate_source="scheduling_v32_v34_qualified.json",
    )
    (HERE / "scheduling_v36_codegen_equivalence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
