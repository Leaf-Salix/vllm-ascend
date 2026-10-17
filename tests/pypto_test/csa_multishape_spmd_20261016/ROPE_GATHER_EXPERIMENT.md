# RoPE相邻交换与逐行gather合并实验

2026-10-17。只在本实验目录修改源码，不改生产入口、冻结静态原型、PyPTO、
Simpler或PTOAS。允许各batch/历史档位分别编译；仍以完整CSA无插桩L1首尾为性能口径。

## 无gather相邻交换CPU探针

背景见`PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md`：当前PTOAS在整个函数含gather/scatter时，
保守地跳过冗余同步消除。本实验不会删除这项安全检查。

- `rope_pair_swap_probe_v1.py`把512个FP32元素reshape为256×2，再extract奇偶列并concat。
  PTOAS拒绝：8字节行宽不满足A3的32字节对齐。未执行设备代码。
- `rope_pair_swap_probe_v2.py`改为64×8，转置后交换四对对齐行，再转回。
  `rope_pair_swap_probe_cpu_v2/result.json`记录完整CPU代码生成通过。
  这需要两次转置及八行复制；不把编译通过当成更快，暂未集成或上卡。
  CPU生成代码没有TGATHER，但这不能证明完整CSA消除gather后能达到700μs。
- 现选QK页表缓存还有真正的非连续gather。只替换RoPE不能使整个AIV函数无gather；
  不会偷偷关闭已验证的QK地址缓存，也不把减少源操作数当作时间收益。

## 三处RoPE逐行gather合并

`rope_flat_gather.py`仅替换以下三个循环：

| 数学函数 | 原循环行数 | 新gather次数 |
| --- | ---: | ---: |
| kv_rms_norm_rope | 32 | 1 |
| rmsnorm_rope_cache_write | 16 | 1 |
| rmsnorm_rope | 16 | 1 |

原第r行的列选择c转换为连续Tile里的`r*64+c`。c仍读取原输入Tile，
没有将任意索引偷偷假定为相邻交换，也没有新增Host读或GM工作区。
整数位移生成行基址，浮点数学和运算顺序不变。原单行tail不改。

AST检查把三个替换块还原后，与原完整catalog完全一致；穷举262144个合法行/列/选择组合，
确认地址仍对应同一行的原选择列。GM读写、FFTS、跨核等待与发布保持原样。
这不替代设备精度、边界和图重放检查。

CPU准备`rope_flat_gather_prepare_v1`通过。首个编译命令误用了`static/`路径，
`rope_flat_gather_cpu_v1`只记录FileNotFoundError，未进入编译；正确目录是`variants/`。
修正后`rope_flat_gather_cpu_v2`在编译。集成开关`--rope-flat-gather`默认关闭，
先与现选QK Vector索引、行依赖及100tick等待组合验证128K/B16/S6。
截至此记录尚无新性能结果，七档选择保持不变。


## 完整CPU编译与设备提交

CPU v2完整PASS；实际1024/1024/2048元素的三个gather索引链、FP32/INT32 Tile与同时活跃UB区间检查通过。集成prepare_v2与已编译源码全部AST相同。详见rope_flat_gather_codegen_audit_cpu_v2.json和rope_flat_gather_integration_audit.json。
任务`task_20261017_131052_534942170`已提交，输出`b16_row_deps_rope_flat_v1`；128K/B16/S6，完整精度/图/30代/保护区后才比较三窗L1。等待设备结果，不预记PASS。


## 完整设备结果：未采用

task_20261017_131052_534942170 exit0；eager/graph/final输出、TopK、六cache-state均零差异，30代与保护区全部PASS。实际三处Tile/索引链/UB与100tick等待门禁PASS。完整L1 819.58/815.10/810.50，中位815.10；同轮动态892.30/923.18/920.58，中位920.58。没有优于现选证据，未采用、不扩档；没有本轮现选配对，不精确归因跨轮差额。源码和原始结果保留，不重复该候选。
详见`rope_flat_gather_result_v1.json`及`b16_row_deps_rope_flat_v1/`。
