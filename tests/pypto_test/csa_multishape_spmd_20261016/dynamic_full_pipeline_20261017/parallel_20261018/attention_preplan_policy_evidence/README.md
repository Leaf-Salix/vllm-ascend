# CSA请求几何按区间前置：共用数学，集中选择时点

2026-10-18。响应性能优先且允许多档位的要求，正式实现位于
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`。不是为七个验收点写白名单。
本轮不修改PyPTO、Simpler或PTOAS，不改共享安装。

## 策略与实现

`policy.py`集中选择 `early_attention_geometry`：

- 已有G6且16<T<32的QA小M16并行区间，启用前置；由于G6已有条件，实际从T24起。
- T>=64启用前置，初始依据为至少两个完整M32投影行块；新增T66边界验证。
- 其余范围在原Top-K/plan位置计算，不读取几何前缀；128K/B8/S6位于此范围。

T取物理token轴，不假定T/B是所有请求的实际query长度。Q/L/position/page
仍从device读取并按本轮内容决定有效范围、地址和尾块。阈值是初始区间策略，
不是理论最优或全范围性能保证；混合长度、其它历史上界和连续轨迹仍需后续验证。

五个正式模块修改：`policy.py`选择策略，`entry.py`贯穿constexpr，
`vector_schedule.py`选择前段调用，`indexer_publish_stages.py`和
`attention_plan_stages.py`在原位置按同一规则读取已准备的几何或现场推导。
`EARLY_ATTENTION_GEOMETRY`不是新的Host数值入参；根Tensor仍130个，
符号轴仍58个，一个MIX SPMD。其它模块及Score/Attention数学保持原实现。

前置版仍借 `attention_indices` 每行八列对齐前缀保存五项metadata。
唯一八行owner在UB写入并MTE3发布，原ROPE phase保护可见性；Top-K读取
后，plan等待该行Top-K owner再覆盖成最终512索引。页保护仍调用同一个函数，
非法页回退、dummy、尾块与浮点边界不省略。后置版完全不读这份暂存。
所有核间发布、获取及环槽READY/ACK保留，不因分档缩减正确性约束。

## 本轮设备数据

五个代表/边界任务均exit0；完整PTO精确输出、Top-K/score、cache/state、
A→B→A、图、保护区和代际检查通过。各行均为同进程同根五窗中位完整L1，
单位μs；含内部等待，不含外围AICPU/orchestrator。正式W8A8第4层、S6、
合成历史。原始窗口与命令在 `RESULTS_V1.json` 和 `SUBMISSIONS_V1.json`。

| 历史/B | T | 前置 | 原多task | 原动态 | 区间候选 | 较原动态降低 |
|---|---:|---|---:|---:|---:|---:|
| 128K/B5 | 30 | 是 | 584.78 | 571.26 | 544.50 | 4.684% |
| 128K/B8 | 48 | 否 | 650.52 | 642.32 | 646.38 | -0.632% |
| 128K/B11 | 66 | 是 | 765.72 | 750.90 | 731.56 | 2.576% |
| 128K/B16 | 96 | 是 | 882.78 | 842.88 | 832.62 | 1.217% |
| 8K/B32 | 192 | 是 | 968.78 | 968.12 | 951.06 | 1.762% |

128K/B8记录0.632%回退，4/5配对窗口更慢；没有删窗口或重复采样取好结果。
直接比较实际被测的AIC/AIV生成C++，仅忽略整行注释，可执行行完全一致。
因此本区间没有给B8增加核内运算，不将4.06μs差值认定为前置开关的因果结果，
也不声称本轮性能改善。保留原数据，不为该差值增加难维护的专门代码。

## 复用已有验证与新增编译核对

`audit_selected_codegen.py`记录14项直接文本比较，不做hash：

- B8关闭路径对同轮原动态；小M16区间T30对先前T24；较大T区间T66/T96对先前T96；
  短历史B32对先前前置版。AIC/AIV可执行C++逐行一致。
- 长历史大T与短历史G2中T补符号编译，比较同一等待hook之前的完整PTOAS C++，
  与已测B24、短B16前置版一致；原B24/B28也是同一动态策略代码。
- 前置/后置、长大T、短G2四次完整CPU编译均通过。CPU不冒充精度/性能证据。

既有首版前置七档及S1/S5、metadata/padding验证在
`../attention_preplan_evidence/`，后置原版在 `../attention_dependency_evidence/`。
本次七档的选择在 `SELECTED_POLICY.json`：六档前置，一档后置。
没有把旧轮次数据拼成一张声称新测七档的表，也没有重复未改变的完整七档。
原多task逐档降低30%的目标仍0/7，当前结果不能替代最终同轮验收。

## 独立计划探针维护

此前行依赖改动给plan helper增加signals/generation，旧可选探针漏接。
同步canonical与本目录的 `plan_device_probe.py`，显式incore边界、准备好状态
的信号Tensor、完整绑定constexpr；探针使用后置几何，不伪造真实生产者时序。
最终v3四组异构Q、dummy、页空洞/越界、短表、陈旧位置及allocation保护通过，
另有两种页前缀列范围检查。它是独立正确性测试，不是性能或完整CSA证据。
v1因测试入口漏初始化runtime失败，v2因依赖调用未显式绑定新增constexpr失败，
两次均留脚本/日志。修复测试适配，没有绕过检查或修改编译器。

## 复现

通过 `task-submit --device auto --max-time 1800` 运行 `run_device.sh`，传新输出
目录及 `--batch B --history H --launches 3 --l1-windows 5`。CPU使用
`run_cpu.sh` 配对应controls JSON；独立计划用 `run_plan_probe.sh`。
所有真机任务必须由队列分配设备。source_candidate/source_reference为冻结源，
完整源、测试与结果入Git；大编译快照、bin/so仅本地保留并登记路径/大小。
