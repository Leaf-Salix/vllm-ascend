# 当前原 cache 路径的 atomic 受控诊断

算子71153bb3，冻结目录`.cache/csa-forward-boundary-71153bb3`。不改算术源码或cache布局。
只切换已存在的`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`。Native控制、mode2、det0、HCCL=false、EPLB关。
QR/KV从跨核split-K atomic加法改为单核固定K顺序，因此此实验同时改变归约次序和分片量，
不能把结果只归因为硬件atomic抖动。

动机：七档GMM任务duration均增加；B8正式forward持续慢，8K/B16两个rank少接受一个草稿。
历史atomic0实验是source-split/ordered缓存和旧分组，未覆盖当前原cache版本的受影响两档。
本次不扩大七档，不先改生产默认值。

先运行[单卡命令](run_layer.sh)：128K/B8和8K/B16，各atomic1/0的20次图重放，
记录本体代价、P95/max和已有保护区/Top-K检查。固定归约全CSA曾在adaptive分支单卡输出状态通过；
此处补两档当前部署形状的性能成本，然后才做真实EP16干预。

是否保留取决于正式forward、token和DSpark共同结果，不以单卡或不同轮profile胜负决定。
