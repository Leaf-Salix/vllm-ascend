# 单因素physical gather静态复核

控制为0a5baa322已验收Vector v7实现。v1模式flag共享缓存线的设计在NPU之前弃用；v2隔离64B后仅INDEX→FP32转换不被固定后端支持；v3通过INT32中转，未改SDK。

- 真实helper AST 480组/245760元素与INT32 Scalar地址oracle相同；Negative sentinel和2**24页上限覆盖。
- baseline/candidate完整64Tensor root编译/Artifact.load，NPU初始化false，manifest、SOURCE、SDK DSO强绑定。
- 独立生成物复核：flag[T,16]，t*16INT32；checked task尾PIPE_ALL→dcci→dsb，checked→plan→qk依赖保留。
- 两处MGATHER均INT32[1,64]→BF16[64,512]，row/zero；索引UB[0,256)，输出[114944,180480)，不与需跨预取保留m/l/oi重叠；AIV最大187072B<192KiB。
- 原MTE3/FFTS事件次序、SWA与逻辑回退不变。该静态结论不代替硬件性能或精度。

## lookahead-v1

Root64Tensor baseline/candidate CPUcompile+Artifact.load通过、NPUfalse，manifest与SOURCE绑定。独立审查源码+生成物：AIC L1 carrier384×512 BF16@0，Q@393216，prob@458752；CMPparts2/3逻辑row128/256跨PV0保留，PV0以及PV1parts0/1仅重载槽0。dynamicPV每个block的part0/K0重新TMATMUL，后续ACC，FIX→M等待保留。AIV score双槽core×128+sb×64+lane+headgroup，早预取后才loadscore，MTE3完成后softmax复用UB。最大187072B；KV临时114944..180480，running状态49152..114944，无生命周期覆盖。抽象协议64mask/64000随机交错通过。不代替NPU精度验收。新增3MiBGM不自动继承128K小ring容量资格。

## direct-publish-v5

两CPU完整64ABI/Artifact.load/NPUfalse、46manifest绑定通过。独立生成审查：carry left[16512,49280)、right[49280,82048)跨四组完整保留；新extract@98432/106624、concat/DIV@82048、swap@114816以后不覆carry。UB最大147584B（144.125KiB）。实际TROWEXPANDDIV→RINT BF16→FP32→2×TCOLEXPANDMUL→TADD→RINT BF16；FP signedSin树与inactive+0保持。li读取原同lane li_transfer，初始化/每有效块写回且MTE3→MTE2/V等待保留。完整query发布后才AIV下一query，无同lane覆盖。编排qk add_output(heads)→兄弟WO-A add_input(heads)实际RAW依赖，旧packed/native_heads已删除。此为代码/生成资格，不等于设备精度或性能。
