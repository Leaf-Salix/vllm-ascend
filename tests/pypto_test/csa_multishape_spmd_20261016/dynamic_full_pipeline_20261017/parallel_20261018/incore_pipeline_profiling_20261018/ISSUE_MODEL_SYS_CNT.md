# incore-profiling：CANN A3模型的SYS_CNT恒零会使计时退避循环无法退出

这是一次模拟环境兼容性报告，涉及incore-profiling的使用边界；当前证据指向
CANN模型的寄存器行为，不能据此归因为PyPTO编译错误。希望skill能记录或
提前检测此限制；如存在启用SYS_CNT建模的配置，也请指出正确用法。

## 环境与已验证范围

- CANN 9.2.0-beta.2，来自同一安装的msprof/msopprof及injection库。
- target=a2a3，SoC=Ascend910_9392，arch=dav-c220，camodel=dav_2201。
- 一次24-block MIX启动，24 AIC+48 AIV；完整CSA的真实物理输入、NZ布局、
  GM别名和device metadata，使用原版incore-profiling脚本及独立输入包装。
- **相同独立kernel.so在真机成功**：77份可写完整storage、756895848字节
  与原调用后的结果逐字节相同，包括输出、Top-K/score、cache/state及保护区。

## 触发代码

当前硬件上的GM信号轮询在未就绪时使用短暂计时退避：

```cpp
while (true) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
  if (*ptr >= expected) break;  // ptr为volatile；writer单写者发布另有DCCI/DSB
  const uint64_t start = get_sys_cnt();
  while ((uint64_t)(get_sys_cnt() - start) < 100) {
    __asm__ __volatile__("" ::: "memory");
  }
}
```

硬件运行完成，但模型中每次SYS_CNT读取均为0，内层循环永远不会返回GM
条件检查。外层即使有spin上限也无济于事，因为它没有机会再次推进。

## 实际模拟证据

开启`msprof op simulator ... --dump=on`后，逐核`instr_log.dump`直接显示：

```text
[00667903] ... MOV_XD_SPR XD:X23=0, SPR:SYS_CNT,
[00667904] ... SUB ... XD:X23=0, XN:X23=0, XM:X22=0,
[00667904] ... JUMPCMP ... XN:X23=0 ... XM:X14=0x64,
...
[00667920] ... MOV_XD_SPR XD:X23=0, SPR:SYS_CNT,
```

AIV也呈现相同行为。对全部72核日志尾部取样，72核均读到SYS_CNT=0，
指令周期持续增长。此前不开dump的运行直到本地25分钟队列时限才被终止；
延长时限并开启dump后定位到该循环，随即主动停止，不将其记成完整采集成功。
同二进制的无profiler模型诊断也未完成，已停止；它本身不提供寄存器归因，
归因依据是上述逐核指令日志。

## 建议的最小排查与处理

可用不含CSA、GM通信或无限循环的独立kernel检查该环境的计时寄存器：

```cpp
uint64_t begin = get_sys_cnt();
for (int i = 0; i < 4096; ++i) {
  __asm__ __volatile__("" ::: "memory");
}
uint64_t end = get_sys_cnt();
out[0] = begin;
out[1] = end;
```

这是建议的最小复现形式，尚未另跑此缩减kernel；已执行的证据来自完整
72核回放。请勿将此代码写成“最小用例已通过验证”。

建议skill文档提示：不能保证硬件计时/特殊寄存器在模型中完整实现；对
依赖计时退出的轮询可做预检查或fail-fast，避免只报长时间采集超时。
不要自动删掉用户通知/等待逻辑或默默修改kernel后声称复现了原时延。

我们目前仅在**单独的模拟诊断副本**为内层增加128次读取上限，保留外层
GM条件、DCCI/volatile、写侧发布、payload fence和72核分工，已完成本用例验证。
该修改改变轮询退避间隔，不能当成原实现的定量等待或完整耗时证据；正式
源码和已验证的原始二进制不变。报告此临时处理不等于请求默认采用它。

另一个独立的timeout单位问题见#1486；它与SYS_CNT恒零不能混为一谈。
本次只提issue和建议，不提交PR，不添加标签。

## 对照：此前HC_pre全核采集正常

进一步核对同一CANN安装路径下此前HC_pre的完整24-block采集：72条核轨迹、
72份指令CSV均已导出，五份输出与真机逐bit相同；全部CSV没有SYS_CNT读取。
HC_pre使用硬件syncall(MIX)、缓存维护和组内事件，没有上述计时退避。
因此本报告不主张模型的多核通信普遍失效；触发范围是当前环境的SYS_CNT
恒零与依赖该寄存器退出的循环。新增有界诊断副本已越过原卡点、出现实际
MMAD，并最终完成下述验证。

补充终态：有界诊断副本的模型计算已完整结束；77份可写storage共756895848
字节与原真机after全部逐字节相同。采集任务已completed、exit=0，原版skill
manifest显示exported，导出72条核trace与72份指令CSV。这验证了本用例的
数学/内存/依赖结果，不能证明替换退避后具有相同的等待时间或性能。

进一步明确诊断边界：读取次数上限未按硬件100tick校准。轮询频率变化
还可能改变DCCI/GM请求、访存竞争、阶段起点与计算/搬运重叠，因此限制
不限于wait区间，不能剔除wait后就将其余流水当作原二进制的定量性能。
源码数学体相同和输出一致也不证明两份二进制非等待部分逐指令相同。
