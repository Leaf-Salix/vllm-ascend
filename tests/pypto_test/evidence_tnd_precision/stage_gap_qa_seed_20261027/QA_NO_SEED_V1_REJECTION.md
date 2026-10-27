# QA no-seed v1 设备前拒绝

首次源码审查以当前formal作比较，实际CPU生成审查发现本轮3399冻结baseline没有T95 packet route，v1复制本地当前formal引入了95路线，故不是严格单因素。v1未设备运行。

v2从远端冻结source读取两文件，generic SHA773f04fd…、packet SHAea30b2de…均匹配binding/测试源manifest；仅删除两处seed，不加入95路线。保留v1和CPU产物作为拒绝证据。正式分支不因此删除已有T95处理；后续正式移植必须保留其原路由，并重新资格验证。
