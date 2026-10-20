# 检查历史

1. 本地K-trace证明先发现s+1为16倍数时是包间环绕，不是K256包内部wrap；修正分段公式，全部256个s的packet起点/宽度及K16次序均通过。未上卡。
2. CPU v1在split的NZ K偏移非负证明失败。有效输入下偏移本来非负，但条件SSA未被证明器追踪；显式max(0,k)表达同一合同，不改SDK/证明器或合法输入地址。保留失败源码及收据，未上卡。

3. CPU v2的max被Simplify消去，后续NZ证明仍不能追踪phi。读实际passes_dump后改成等价的wrap_base=240+base%16及0起点post循环，避免让K地址依赖条件phi；重新穷举256个起点的packet-width序列。未修改SDK。

4. CPU v3三个候选均在L0C控制流合并失败。v4保持packet顺序，split取消包裹Acc的if改为0次数循环；K128将GM assemble放进各分支，避免Acc跨分支逃逸。仍无SDK改动。

5. CPU v4仍因L0C合并失败，不占卡。新独立nopipe仅把父版seed_fast pipeline换range；k128narrow在aligned且same-start快路径用K128，其余沿原M16回退，避免新增M128 Acc if-phi。生成脚本一次缩进边界提取失败，在本地syntax门禁已拦截，未提交NPU。

6. CPU v6发现显式Mat转换把tensor.slice(shape,offset)当作load(offset,shape)，导致静态tile shape变动态。已修正参数顺序，保留失败源码；未上卡/未改SDK。

7. DFX v1在设备初始化前因调用不存在的torch.DfxConfig失败；已读当前SDK实际init签名，恢复前轮验证过的enable_chip_swimlane=1/enable_dep_gen=False/output_dir接口，并新增无设备signature.bind门禁。原始失败脚本及task日志保留。

8. PMU v1已完成全部9次采样及70项逐位比较，但残留正式计时footer引用samples而NameError退出。原始结果与CSV保留，标为采样完成/任务失败，不冒认任务通过；修复后对全部新probe做F821未定义名检查。DFX重提同时去除同一残留footer。

9. DFX v2因保留原无参数init后又配置DFX，产生Worker配置冲突。已对照前轮成功源码，在原位置只保留一次配置init；新增AST单次初始化检查。未出现内核数值差异。

10. v9权重Mat切片曾存在父NZ步幅丢失疑虑，实际生成TEXTRACT从完整父Mat取行，独立审查解除阻断后才上卡；该候选精度通过但性能回退2.80%，不推广。

11. 本地离线统计初稿按`-t`拆文件名误截tail候选，已改为完整正则；PMU离线验证初稿生成器括号遗漏已修正。都未触发NPU任务，raw未修改。

12. M256 stage2 v10无设备编译被Left131072B>65536B容量检查拦截。未改SDK；新建v11 M128/M256两臂同K128 range控制，完整编译和实际生成码通过后才上卡，不将行高+取消预取的复合改变冒充对stage2的单因素。
