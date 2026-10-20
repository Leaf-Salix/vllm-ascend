# 本轮检查历史

1. 纯等待初版wait24按0..23，与wait8首地址不同，可能混入热点顺序变化。保留原源码/CPUv1但不上卡，另建wait24aligned，前8个地址和顺序与wait8完全一致，CPUv2完整通过后用于主对照。
2. 两MIX初稿phasecut删等待时也删了payload acquire，独立设计审查指出跨task完成不替代缓存获取。初稿仅CPUv1编译留档，未上卡；新phasecut-acquire保留原条件下fence/cacheinvalid ALL/fence，再cast。
3. 切点CPUv2 checker因注册名含连字符被拒绝，未编译内核、未上卡。v3只把checker注册名转换为下划线，同一候选源码完成PTOAS/CCEC及artifact.load；原失败回执保留。
4. 生成编排文件名为orchestration/resident_suffix.cpp，初次按main.cpp读取失败后已按实际文件定位；归档包括真实编排，不以cluster名证明切点。

5. PMU JSON下载的自动审批未在deadline内完成，工具明确并非安全风险判定；按允许唯一重试后成功，没有重交任务。
6. 量化切点初稿的无Cube算术函数被分类为AIV，同cluster两个AIV实际注册仅首个空闲入口。实际kernel_config检查在设备前阻断；新建显式单AIV48源，逻辑worker0..47、槽24..71不变，完整编译/实际角色/runtime审查后上卡。没有修改SDK，也没有将初稿错误当设备精度失败。
7. 本地默认Python3.9不支持tar.extractall(filter=...)，原始包未改；改用明确的workspace Python并先拒绝绝对路径、..、链接/特殊成员再解包。本地校验失败未触发NPU任务。
