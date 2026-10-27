# 本轮整模型候选身份说明

- 正式已验收计算控制为11b380e3fc16efdc1cb85a0b2888720872c70655；旧验收SDK为PyPTO2dfa7f12/PTOAS0.66，旧环境保留。
- `qa-whole-formal-v1`派生于已封存Direct正式whole助手，仅generic及packet QA的seed增加`ATOMIC_ADD or QR_OK != 1`保护，95/126/168/generic分派与计算AST保持；独立gap_generic，不覆盖冻结snapshot。
- 原13包静态资格、Native gold、bank、16rank token及接受事件、调度入场、zero-preempt与容量检查保留。
- 当前有效CPU收据为该候选的`cpu-tuned/receipt.json`，helper manifest为7e7f53ae…、SOURCE为8778635e…；root64、NPUfalse、1～256路由与注册均通过。外部归档`qa-whole-cpu.tar.gz`同时保存8K无设备preflight，Native源与gold一致。
- SOURCE里baseline_commit/checkpoint/actual_control和FORMAL_CPU_RECEIPT.json是继承的历史谱系，不能当本轮控制、当前编译或本轮硬件通过证据。冻结文件不事后篡改。
- 新SDK单层性能和Native逐bit结果不是本轮旧SDK整模型token验收。只有实际whole任务与逐rank审计成功才能报告本候选token通过。
