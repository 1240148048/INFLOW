# 开放式混合实验验证记录

日期：2026-11-06。

2026-11-07补充：统一四模型入口与架构配置适配后，相关回归测试13项通过、1项真实Linux执行测试在Windows跳过。覆盖五种config类型（含Qwen3.5组合/文本配置）、共享协议不被修改、Qwen3.5真实小型checkpoint权重加载及LoRA目标、训练恢复和分批生成。尚未执行四个正式骨干的GPU实验。正式AutoDL入口统一为 `scripts/run_autodl_open_500.sh`，以MODEL切换骨干。

AutoDL补充：新增显式process执行后端，新增3项调度/环境/超时测试通过；真实Linux子进程执行测试在本Windows环境跳过，需按AUTODL.md在服务器运行。AutoDL配置不要求嵌套Docker，隔离强度不等同Docker。

已验证：

- 新任务与相关KL测试共23项通过，包括真实小型causal LM + PEFT的生成/表示、student重新采样、相同前缀、teacher梯度冻结，以及中断后EMA/optimizer/数据顺序恢复。
- 从真实公开数据成功准备并合并：TOPv2 150/150、MBPP 150/150、CoQA 100/100、SQuAD 100/100；最终精确500训练题、500测试题。
- 训练文件不含目标gold答案/parse/参考代码/测试断言；训练与测试ID不重叠；CoQA历史依赖均指向已出现的前序turn。
- 150道TOPv2参考树均能解析；MBPP仅150道测试题的官方断言独立保存。
- 五个Bash脚本通过语法解析，Python模块通过编译检查。

验证环境：Windows，PyTorch 2.12.1+cpu，Transformers 5.12.1，PEFT 0.21.2。

未在此环境执行：正式Qwen GPU生成/训练、Docker镜像构建、MBPP容器程序执行。没有生成或声称任何正式准确率。运行入口为 `bash scripts/run_open_mixed_500.sh`，正式实验应在Linux CUDA/Docker服务器完成。

数据加载、小模型集成测试与语法检查不能替代正式实验结果。各指标和CoQA单参考/模型历史协议见OPEN_TASKS.md。
