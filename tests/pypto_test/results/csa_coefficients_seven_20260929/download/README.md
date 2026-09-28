# 七档JSON下载包

21份原始JSON：每档Native/PTO各一份PyTorch profiler导出，另含PTO的第4个独立泳道窗口。
窗口编号预先固定为window_3，不选择最快窗口；四个完整窗口仍保留在原目录。
源码4ffccb7b，CANN9.2/mode2/det0，PTO atomic0，正式layer4权重及独立合成历史。
范围为HC_pre+norm+CSA+HC_post，不能当作真实权重16卡模型forward或token验收。
性能主表来自独立5预热/20次无profiler事件，不能直接用这些独立profile做相减归因。

[性能结果](RESULTS.md)；[原文件映射](sources.json)。JSON只复制，未修改或拼接事件。
