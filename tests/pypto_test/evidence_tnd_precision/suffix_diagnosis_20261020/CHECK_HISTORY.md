# 离线检查历史

- DFX初次离线检查误把每root开始标记当作起止对；读取实际SDK schema后按真实root数检查，原始数据保留，未修改硬件数据。
- 下载了仍running的incremental T126中间result.json。完整结果校验正确拒绝pass=false/complete=false的快照；快照按原样保存在inprogress-incremental-t126-v1-20261020.json。等任务结束后另取最终结果，不把未完成快照当作kernel精度失败。
