# QK/PV AIV内PIPE_ALL来源核对（2026-10-17）

使用当前815.84μs的128K/B16/S6配置生成的原PTO IR，只运行已安装
PTOAS 0.66的同步诊断，没有修改编译器、ISA或依赖仓。

## 结论

原PyPTO IR的QK tick内没有这四个显式PIPE_ALL。实际PTOAS日志对照
RemoveRedundantSync之后、EventId Allocation之后的同一依赖编号，确认
部分普通set/wait依赖在事件分配时回退为全pipe屏障。

|依赖编号|分配前的方向|落点|分配后|
|---|---|---|---|
|2708|V→MTE2|KV段后/softmax段前|PIPE_ALL|
|2709|V→MTE2|softmax分支的Score加载|PIPE_ALL|
|2712|V→MTE2|PV合并分支入口|PIPE_ALL|
|2687、2688、2689|MTE3→V|PV合并分支入口|PIPE_ALL，邻近屏障会合并|
|2714|MTE3→V|最终输出的concat|PIPE_ALL|

编号只对应本次保存的IR，不是稳定API。前四个中的两个屏障位于此前
四段SYS_CNT计时区间外；另外两个位于softmax与最终输出内部。
不能把四段计时差额全部归因于屏障，也不能把静态屏障数当运行次数或耗时。
这些同步包含真实的片上缓冲读写依赖，不能直接删除。

将valid_block_mask预读到本核UB的候选已经通过CPU生成，但屏障仍是4处；
因此“GM标量读直接导致这四个屏障”的猜测没有成立。mask候选没有上卡测性能。

## 可追溯证据

- `qk_sync_diagnostic_v4/sync.log`：完整已安装PTOAS的诊断与pass后IR。
- `qk_sync_diagnostic_v4/qk_dependency_nodes.json`：对应生产者/消费者节点。
- `qk_sync_diagnostic_v4/qk_fallbacks.json`：同一编号的前后记录。
- `qk_mask_prepare_v2/codegen_sync_comparison.json`：mask缓存前后代码计数。
- `qk_phase_probe_cpu_v1/uncovered_sync.txt`：分段计时外的同步位置。

可复现命令（D为本实验worktree，N为本目录，输出另选空目录）：

```bash
/data/pyptouser/qinchuanyu/pto-eager/.cache/dsv4-toolchain/ptoas-0.66/bin/ptoas \
  "$N/b16_cached_qk_indices_v1/build_output/entry_57ys1c05/ptoas/csa_static_group.pto" \
  --enable-insert-sync --pto-level=level3 --pto-insert-sync-debug=2 \
  --mlir-print-ir-after=pto-insert-sync -o "$OUT/group.cpp" \
  > "$OUT/stdout.log" 2> "$OUT/sync.log"
```

v1/v2的打印参数失败、v3调试日志过大而主动终止，均记在ATTEMPTS.md；
最终结论只来自完整结束的v4。大日志保留本地并记路径，不做hash。

MGATHER和HC 2行分片随后各通过整层exact门槛，但未优于815.84μs，
默认配置不变。详见`qk_mgather_result.json`、`hc_rows2_result.json`。
