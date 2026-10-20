# SPDX-License-Identifier: Apache-2.0
"""检查生成的 SDMA 入口及业务 Vec 分配根；动态子视图不误当作新的分配。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cpp_path = args.kernel / "kernels/aiv/vector_schedule.cpp"
cpp = cpp_path.read_text()
body = cpp[cpp.index("static __aicore__ void vector_schedule("):]
constants = {name: int(value) for name, value in re.findall(
    r"const (?:u?int(?:32|64)_t) (\w+) = (-?\d+);", body
)}
# TASSIGN 的地址常经过一次 index→uint64 转换；只追踪单次赋值的纯别名。
# 不把可能在循环中变化的普通临时量当成常量。
assign_counts = Counter(re.findall(r"^\s+(\w+) = ", body, re.MULTILINE))
aliases = re.findall(
    r"^\s+(\w+) = (?:\((?:u?int(?:32|64)_t)\) )*(\w+);$", body, re.MULTILINE
)
while True:
    before = len(constants)
    for name, source in aliases:
        if assign_counts[name] == 1 and source in constants:
            constants[name] = constants[source]
    if len(constants) == before:
        break
vec = set(re.findall(r"Tile<TileType::Vec,[^\n]+> (\w+);", body))
addresses = {}
dynamic_views = []
for name, address in re.findall(r"\bTASSIGN\((\w+), (\w+)\);", body):
    if name in vec:
        if address not in constants:
            dynamic_views.append({"tile": name, "expression": address})
        else:
            addresses[name] = constants[address]
assert addresses
# 行 gather 的子 Tile 在 C++ 中是 base + 动态行号，不能假装它是静态地址。
# 改查生成 PTO 的所有 alloc_tile 分配根，仍要求每个根的真实地址可解。
mlir = (args.kernel / "ptoas/dynamic_csa_full.pto").read_text()
parts = re.split(r"(?=\bfunc\.func(?:\s+\w+)?\s+@)", mlir)
sections = [part for part in parts if re.match(
    r"func\.func(?:\s+\w+)?\s+@vector_schedule\(", part
)]
assert len(sections) == 1
vector_ir = sections[0]
ir_constants = {name: int(value) for name, value in re.findall(
    r"(%[\w]+) = arith\.constant (-?\d+) : (?:i64|index)", vector_ir
)}
allocation_bases = []
for line in vector_ir.splitlines():
    if "pto.alloc_tile " in line and "loc=vec," in line:
        match = re.search(r"addr = (%\w+)", line)
        assert match and match.group(1) in ir_constants, line
        allocation_bases.append(ir_constants[match.group(1)])
assert allocation_bases
context = re.findall(r"= pypto_make_prefetch_context\(\w+, \w+, (\w+)\);", body)
assert len(context) == 1, context
scratch = constants[context[0]]
assert scratch == 0
assert all(base >= scratch + 256 for base in addresses.values())
assert all(base >= scratch + 256 for base in allocation_bases)
assert "get_sdma_channel_id(args)" in cpp
assert "interval = PLATFORM_PROF_SYS_CNT_FREQ / 1000000ULL" in cpp
assert "timeout = PLATFORM_PROF_SYS_CNT_FREQ / 50ULL" in cpp
assert body.count("= TPREFETCH_ASYNC(") == 1
assert body.count("= pypto_wait_prefetch(") == 1
assert "GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 8388608>" in body
result = {
    "passed": True,
    "generated_cpp": str(cpp_path),
    "scratch_range": [scratch, scratch + 256],
    "business_vec_allocation_count": len(allocation_bases),
    "minimum_business_vec_allocation_address": min(allocation_bases),
    "maximum_business_vec_allocation_address": max(allocation_bases),
    "scratch_disjoint_from_all_business_allocations": True,
    "cpp_constant_tile_addresses_checked": len(addresses),
    "cpp_dynamic_subviews_not_statically_evaluated": dynamic_views,
    "resident_channel_used": True,
    "paced_wait_used": True,
    "static_prefetch_call_sites": 1,
    "physical_prefetch_region_bytes": 8388608,
    "device_validation": False,
}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
