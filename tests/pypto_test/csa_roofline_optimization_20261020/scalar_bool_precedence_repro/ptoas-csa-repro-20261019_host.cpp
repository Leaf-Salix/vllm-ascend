#include <cstdio>
bool mixed_bool(bool v1, bool v2, bool v3, bool v4, bool v5) {
  using T = float;
  // pto: %ab, %cd, %x, %y
  return (v1 & v2 & v3 | v4) & v5;
}
int main() {
 int bad=0;
 for (int i=0;i<32;++i) {
  bool a=(i&1)!=0,b=(i&2)!=0,c=(i&4)!=0,d=(i&8)!=0,e=(i&16)!=0;
  bool expected=(a&&b&&(c||d)&&e),actual=mixed_bool(a,b,c,d,e);
  if (expected!=actual) { ++bad; std::printf("%d %d %d %d %d expected=%d actual=%d\n",a,b,c,d,e,expected,actual); }
 }
 std::printf("mismatches=%d\n",bad);
 return bad ? 1 : 0;
}
