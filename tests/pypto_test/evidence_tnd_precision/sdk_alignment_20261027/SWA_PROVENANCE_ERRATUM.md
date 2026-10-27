# SWA复测谱系更正

冻结helper swa-softmax128-v1的SOURCE.json中“earlier Cube-only SWA128”描述错误。
20261023/swa128已包含同类Vector128 fast path，旧基准上约0.761%收益、12/12领先，八边界精度通过。
本次不是新算法，属于在正式11b380 Direct heads基准上的单因素组合复测，仍使用已验收2dfa/0.66 SDK。
不能因新独立d449/914工具链同时在构建，就将本候选描述成新SDK结果。
该笔误不改变被测数学或控制身份；冻结helper/CPU收据不回写，以保存实际受测SHA。
