# Attention AIC 两个 L1 槽：仅源码准备

2026-10-18。基准直接读取正式提交`66882634`的`attention_stream_stages.py`，没有从其它候选叠加。**供后续Score/Attention同核交错准备，不是独立性能候选；尚未PyPTO编译、上卡或提交测量。**只写本私有目录，正式源、依赖和共享安装均未修改。

文件：`source_reference/attention_stream_stages.py`为正式原文；`source_candidate/attention_stream_stages.py`为候选；`candidate.patch`为完整文本差异；`source_audit.json`保存标准库检查结果。源码现已冻结。

执行逻辑仅四项：新增`QK_L1_SLOTS=2`；AIC的KV Mat容量改为两个128行槽；两处L1行偏移改用`work%2`；每个tick把完整PV(tick−2)块放在完整QK(tick)块前。**所有GM槽仍用原`QK_TRANSFER_SLOTS=3`，AIV从装饰器至函数末逐字节不变**。没有新增输入、工作区、事件、phase、Shape特化或其它优化。

安全的源级依据：r与r−2同属一个L1槽，因此必须先读取旧PV，再覆盖为新QK。首两tick只有QK，尾两tick只有PV，循环仍来自真实T。query切换时旧PV不需要Q，Q只在新query首块更新。原device mask、块地址、store有效范围及每个块内事件条件都保留；失效块不发/不等对应事件，也不读旧槽。GM三槽及AIV mi/li先写后读的原时序不改，避免错误地把mi/li(r−1)和mi/li(r−3)塞进同一个两槽环。

这里没有宣称物理覆盖已验证：PV extract的MTE1读取必须完成后，新KV gather的MTE2才可覆盖同一L1槽；需后端实际事件证明。全局的QK/PV消息发出顺序被有意交换，不能把“每个块事件语句不变”写成完整事件序列不变。缩槽会推迟本轮QK/SCORE_READY，可能损失原先QK先行重叠；本轮没有性能结论。Score Query驻Mat重TMOV及其它交错状态尚未接入。

检查已通过：AIV原文相同；入口签名相同；五个GM地址赋值AST相同；QK/PV子树仅在归一化L1模数后完全相同；整模块去docstring后恰等于“新增常量、三处容量/模数替换、两块移动”的预期AST；真实T循环界、mask条件保持；Python语法通过。没有导入PyPTO、没有运行lowering或设备。

可复现：

```bash
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_two_l1_candidate_20261018/prepare_and_audit.py
```

脚本从上述Git提交读原文，已有源码若与预期不同会拒绝覆盖；不做hash。
