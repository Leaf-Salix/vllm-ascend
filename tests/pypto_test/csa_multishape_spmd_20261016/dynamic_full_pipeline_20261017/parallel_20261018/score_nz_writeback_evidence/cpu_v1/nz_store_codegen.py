# SPDX-License-Identifier: Apache-2.0
"""临时硬件能力实验：只替换Score交接区的完整64×16 FP32 Acc写回。

PyPTO目前把NZ Tensor限定为只读。本hook不是正式算子接口，也不改共享
编译器；先以完整CSA确认PTO-ISA NZ写回是否有收益。匹配同时要求GM根名、
64×16分区和Acc源Tile；其它store、数学、事件、地址基指针均保持。
每个panel独立NZ/C0=8，物理占用仍为64×16个FP32，配套AIV按字节布局读取。
"""

import re


HELPER = r"""
template <typename Global, typename Acc>
static AICORE inline void CSA_SCORE_STORE_NZ(Global& destination, Acc& source) {
  using NzShape = pto::Shape<1, 2, 4, 16, 8>;
  using NzStride = pto::Stride<1024, 512, 128, 8, 1>;
  // Each panel is one complete 64x16 FP32 NZ block; no partial or strided store.
  pto::GlobalTensor<float, NzShape, NzStride, pto::Layout::NZ> panel(
      destination.data(), NzShape(), NzStride());
  pto::TSTORE(panel, source);
}
"""


def rewrite(source):
    """原正式query-major参考没有64×16 Score store，保持逐字不变。"""
    parts = source.split("AICORE void vector_schedule(", 1)
    if len(parts) != 2:
        raise ValueError("完整动态CSA的AIV定义不唯一")
    cube, vector = parts
    definitions = re.compile(
        r"^\s*(v\d+) = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 16>,[^\n]+pto::Layout::ND>\([^\n]+$", re.M
    )
    targets = set()
    for match in definitions.finditer(cube):
        context = cube[max(0, match.start() - 180):match.start()]
        if "indexer_transfer__ssa" in context:
            targets.add(match.group(1))
    acc_tiles = set(re.findall(r"Tile<TileType::Acc, float, 64, 16,[^\n]+>\s+(v\d+);", cube))
    changes = []

    def replace(match):
        indent, destination, accumulator = match.groups()
        if destination not in targets:
            return match.group(0)
        if accumulator not in acc_tiles:
            raise ValueError(f"Score store源不是完整64×16 FP32 Acc：{accumulator}")
        changes.append({"destination": destination, "accumulator": accumulator})
        return f"{indent}CSA_SCORE_STORE_NZ({destination}, {accumulator});"

    cube = re.sub(r"^(\s*)TSTORE\((v\d+), (v\d+)\);$", replace, cube, flags=re.M)
    if targets and len(changes) != 16:
        raise ValueError(f"G6/N64/1024应有16处写回，实际匹配{len(changes)}")
    if not targets and "key_major_inline" in source:
        raise ValueError("发现key-major消费者却没有匹配NZ生产者")
    if changes:
        marker = "AICORE void cube_schedule("
        if cube.count(marker) != 1:
            raise ValueError("AIC定义不唯一")
        cube = cube.replace(marker, HELPER + "\n" + marker)
    result = cube + "AICORE void vector_schedule(" + vector
    assert (result.split("AICORE void vector_schedule(", 1)[1] == vector)
    return result, {"experimental_nz_stores": len(changes), "sites": changes,
                    "per_panel_shape": [1, 2, 4, 16, 8], "aiv_source_unchanged_by_hook": True,
                    "other_stores_and_events_unchanged": True, "production_api": False}
