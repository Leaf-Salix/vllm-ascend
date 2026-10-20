// SPDX-License-Identifier: Apache-2.0
// 与 A3 probe 的逆序 payload 场景对应，核对 CPU-SIM 真实 TSORT32 实现。
#include <cstdint>
#include <pto/pto-inst.hpp>
#include <cstdio>
#include <cstring>

int main() {
  using namespace pto;
  using Values = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor>;
  using Indices = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor>;
  using Pairs = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor>;
  NPU_MEMORY_INIT();
  Values src;
  Indices idx;
  Pairs dst;
  TASSIGN(src, 0);
  TASSIGN(idx, 128);
  TASSIGN(dst, 256);
  for (int j = 0; j < 32; ++j) {
    src.data()[j] = 1.0f;
    idx.data()[j] = 31 - j;
  }
  TSORT32(dst, src, idx);
  std::printf("{\"cpu_indices\":[");
  for (int j = 0; j < 32; ++j) {
    uint32_t payload = 0;
    std::memcpy(&payload, &dst.data()[2 * j + 1], sizeof(payload));
    std::printf("%s%u", j ? "," : "", payload);
  }
  std::printf("]}\n");
}
