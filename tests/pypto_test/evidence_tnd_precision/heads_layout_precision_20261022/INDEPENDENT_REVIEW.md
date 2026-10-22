# 独立原始证据复核

独立审查直接读取原始tar及解包文件，不以父线程RECHECK为结论依据。

- 24份文本文件与tar成员逐字节一致，15文件helper manifest闭合。
- 四份候选、源码绑定、环境与原编译收据均与上轮性能测试同源，canonical3399/77/QA身份一致。
- 四arm按Native、冻结、Direct、Packed完成，PTO namespace/64Tensor/T144/FULL及metadata边界正确，无回退。
- 三PTO arm的A/B对Native共48项边界逐位一致，different/max_abs/relative_l2全部为0。
- compiled/raw、毒化恢复、ABA、33guard通过；额外12次重放共96项状态及396项guard通过。
- readonly权重包在每次重放受实际助手断言保护，不只比较最终输出。
- timing_executed=false，本次没有新性能测量。
- completed exit0，设备8已释放；入口11:52:05～11:56:22，257秒。

结论：本轮两个布局候选的单层128K/B24精度通过，与原8K单层结果共同保留。
不扩大为整模型token、短query/混合TND的新资格，也不代表默认实现已切换。
