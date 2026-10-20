#include <cstdio>
#include <cstdint>
bool mixed_bool(bool v1, bool v2, bool v3, bool v4, bool v5) {
  using T = float;
  // pto: %cd
  bool v6 = v3 | v4;
  // pto: %ab, %x, %y
  return (v1 & v2 & v6) & v5;
}

int32_t mixed_bits(int32_t v1, int32_t v2, int32_t v3, int32_t v4) {
  using T = float;
  // pto: %bc
  uint32_t v5 = (uint32_t) v2 | (uint32_t) v3;
  // pto: %bc, %bcd
  uint32_t v6 = (uint32_t) ((int32_t) v5) ^ (uint32_t) v4;
  // pto: %result, %bcd
  return (int32_t) ((uint32_t) v1 & (uint32_t) ((int32_t) v6));
}
int main() {
  int bad = 0, cases = 0;
  for (int i=0;i<32;++i) {
    bool a=(i&1)!=0,b=(i&2)!=0,c=(i&4)!=0,d=(i&8)!=0,e=(i&16)!=0;
    bool expected=(a&&b&&(c||d)&&e), actual=mixed_bool(a,b,c,d,e);
    ++cases; if (expected!=actual) ++bad;
  }
  uint32_t vals[] = {0,1,3,0x80000000U,0xffffffffU};
  for (uint32_t a:vals) for (uint32_t b:vals) for (uint32_t c:vals) for (uint32_t d:vals) {
    uint32_t expected=a&((b|c)^d);
    uint32_t actual=static_cast<uint32_t>(mixed_bits(static_cast<int32_t>(a),static_cast<int32_t>(b),static_cast<int32_t>(c),static_cast<int32_t>(d)));
    ++cases; if (expected!=actual) ++bad;
  }
  std::printf("cases=%d mismatches=%d\n",cases,bad);
  return bad ? 1 : 0;
}
