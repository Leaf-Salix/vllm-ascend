# SPDX-License-Identifier: Apache-2.0
"""只替换 Indexer WS 的 FP16 NZ→L0B 搬运，用于评估 load2d 转置路径。

背景：原 TEXTRACT 对这一布局调用 img2colv2_cbuf_to_cb；硬件采样显示
Score 的 MTE1 活跃时间较长。本实验保持 PyPTO 图、物理地址、双缓冲、
矩阵运算、FP16 舍入、GM 接口和所有事件完全不变，仅替换这一个搬运实现。

源 NZ 的第 (k1,n1) 个 16×16 分形位于 n1*K1+k1；目标 L0B 的分形
位于 k1*N1+n1，分形内部也转置。每个 n1 发一条 load2d_transpose，
读取 K1 个连续分形，目标分形间隔 N1-1。这是布局转换，没有浮点计算。
这里只接受完整、零偏移、FP16 的同形 Tile，不扩展动态 valid 或尾窗语义。

这是测试侧编译实验；验证收益后再考虑在独立依赖工作区实现通用 lowering。
不能将本文件称为已经合入正式算子或上游的优化。
"""

import contextlib
import json
import re
from pathlib import Path


HELPER = r'''
#if defined(__DAV_CUBE__)
template <typename Dst, typename Src>
__tf__ AICORE void CSA_WS_LOAD2D_IMPL(
    typename Dst::TileDType __out__ dst, typename Src::TileDType __in__ src) {
  // 源 NZ [K,N]：n1*K1+k1；目标 L0B ZN：k1*N1+n1。
  // 每个 16×16 分形内部由硬件完成转置，数值与原 load3d 完全相同。
  auto dst_ptr = (__cb__ half*)__cce_get_tile_ptr(dst);
  auto src_ptr = (__cbuf__ half*)__cce_get_tile_ptr(src);
  constexpr uint16_t k_blocks = Dst::Rows / 16;
  constexpr uint16_t n_blocks = Dst::Cols / 16;
  for (uint16_t n = 0; n < n_blocks; ++n) {
    load_cbuf_to_cb_transpose(dst_ptr + n * 256, src_ptr,
      n * k_blocks, static_cast<uint8_t>(k_blocks), 1, n_blocks - 1, false, 0);
  }
}
template <typename Dst, typename Src>
static AICORE inline void CSA_WS_LOAD2D(Dst& dst, Src& src, uint16_t row, uint16_t col) {
  static_assert(std::is_same_v<typename Dst::DType, half> &&
                std::is_same_v<typename Src::DType, half>, "只支持 FP16");
  static_assert(Dst::Loc == TileType::Right && Src::Loc == TileType::Mat,
                "只支持 L1 到 L0B");
  static_assert(Dst::isRowMajor && !Src::isRowMajor &&
                Dst::SFractal == SLayout::ColMajor && Src::SFractal == SLayout::RowMajor,
                "必须是 NZ 到 L0B ZN");
  static_assert(Dst::Rows == Src::Rows && Dst::Cols == Src::Cols &&
                Dst::Rows % 16 == 0 && Dst::Cols % 16 == 0 && Dst::Rows / 16 <= 255,
                "本实验只支持完整同形分形窗口");
  PTO_ASSERT(row == 0 && col == 0, "本实验不支持窗口偏移");
  CSA_WS_LOAD2D_IMPL<Dst, Src>(dst.data(), src.data());
}
#endif
'''


def rewrite(source):
    """按语义名称和实际 Tile 类型双重限定，不替换其它 TEXTRACT。"""
    cube = re.search(r"(?:AICORE|__aicore__) void (?:score_cube|cube_schedule)\(", source)
    vector = re.search(r"(?:AICORE|__aicore__) void (?:score_vector|vector_schedule)\(", source)
    if cube is None or vector is None or cube.start() >= vector.start():
        raise ValueError("不能唯一确定 AIC 函数边界")
    prefix, body, suffix = source[:cube.start()], source[cube.start():vector.start()], source[vector.start():]
    declaration = re.compile(
        r"// pto: %(buf_(?:previous|last)_pair[^\n]*)\n(?:\s*;\n)?\s*"
        r"Tile<TileType::Right, half, (\d+), (\d+),[^\n]*?> (\w+) =")
    targets = {}
    for match in declaration.finditer(body):
        name, rows, cols, variable = match.groups()
        targets[variable] = dict(name=name, rows=int(rows), cols=int(cols))
    if not targets:
        raise ValueError("没有找到 Indexer WS FP16 Right，拒绝无效实验")
    sites = []

    def replace(match):
        destination = match[2]
        if destination not in targets:
            return match[0]
        spec = targets[destination]
        if spec["rows"] % 64 or spec["cols"] != 64:
            raise ValueError(f"本轮只验证 N64 的完整 WS：{spec}")
        sites.append(dict(**spec, call=match[0].strip()))
        return match[0].replace("TEXTRACT(", "CSA_WS_LOAD2D(", 1)

    changed = prefix + re.sub(r"^(\s*)TEXTRACT\((\w+), ([^\n]+)\);$", replace, body, flags=re.M) + suffix
    if len(sites) != 16 or len(sites) != len(targets):
        raise ValueError(f"应替换本轮 Score 的 16 个面板，实际 {len(sites)} / {len(targets)}")
    marker = "using namespace pto;"
    if changed.count(marker) != 1:
        raise ValueError("helper 插入点不唯一")
    changed = changed.replace(marker, marker + "\n" + HELPER, 1)
    # 去掉 helper、还原调用后必须逐字符回到原代码，包括原 set/wait 顺序。
    restored = changed.replace("\n" + HELPER, "", 1).replace("CSA_WS_LOAD2D(", "TEXTRACT(")
    if restored != source:
        raise AssertionError("搬运替换外出现其它代码变化")
    return changed, dict(sites=sites, count=len(sites), original_restored_exact=True,
                         math_and_events_unchanged=True, device_validated=False)


@contextlib.contextmanager
def patch_score(output, *, unit="csa_indexer_score_profile"):
    from pypto.backend import pto_backend

    previous = pto_backend._compile_pto_module
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def compile_with_load2d(pto_code, unit_name, output_dir, *args, **kwargs):
        source = previous(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != unit:
            return source
        modified, audit = rewrite(source)
        audit["unit"] = unit_name
        (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        (output / "original.cpp").write_text(source)
        (output / "modified.cpp").write_text(modified)
        return modified

    pto_backend._compile_pto_module = compile_with_load2d
    try:
        yield
    finally:
        pto_backend._compile_pto_module = previous
