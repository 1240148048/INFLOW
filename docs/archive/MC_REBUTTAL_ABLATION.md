# 原选择题消融补充包

## 已有 subsets：补齐十个 checkpoint 的新入口

补充消融折线图使用 `bash scripts/start_rebuttal_mc_curve.sh`，详见
`MC_ABLATION_CURVES.md`。该入口直接复用已有 subsets，不重复生成和选样，
训练到 500 updates，每 50 updates 评测一次，并自动绘图。
以下旧入口是此前的 250 步单点流程，不用于本次补充折线图。

## 旧生成缓存也丢失时直接启动

将图3数值CSV放在 `/root/autodl-tmp/figure3_digitized_estimates.csv`，下载Ministral权重到 `/root/autodl-tmp/models/Ministral-3-8B-Instruct-2512` 后执行：

```bash
cd /root/INFLOW_mc_rebuttal_ablation
bash scripts/start_rebuttal_mc_ablation.sh
```

该入口执行run，会下载缺失的SciKnowEval/MMLU-Pro源文件、按固定种子构建500/500题池、重建缺失的候选与1/5教师生成缓存，再训练新增五个消融变体。已有曲线从CSV导入。若保留了原pool/test文件，用DATA_FILE指定可直接复用。重新抽取的数据池可能与旧图的数据池不同，应在汇总时保留这一实验范围区别。

ACTION=import只导入曲线与生成表格；上面的start入口才会启动生成和训练。终点表只显示数值，来源记录位于独立的JSON文件。

数据使用图3对应的原始 SciKnowEval + MMLU-Pro pool/test 文件。模型使用原 Ministral-3-8B 权重和同一 JSON 配置。新增实验默认训练到250次更新，只评终点，每题3次 rollout。学习率调度沿用原图的500步总时域，在250步停止，保证与图上原250步对照一致。若旧图实际调度总时域不同，用TRAINING_UPDATES指定真实值。UPDATES=500可另开目录跑到原图终点。

## 已有曲线原样复用

图中的 Ours、w/o Certainty、w/o Similarity、Random 20% 和 Original 不进入新增训练队列。导入器优先读取旧运行的逐检查点 metrics.json，也能读取旧绘图脚本导出的module_ablation_results.csv。它把fraction单位的Avg@3和rollout-index SD乘100后格式化成两位小数，同时保留来源路径，导入50至500步的全部已有点。

本地仅找到PNG和绘图脚本，未找到该图的原始逐点JSON或CSV。截图的像素分辨率不足以恢复两位小数的精确均值与SD。缺失项保留空白和 missing_raw_metrics 标记。Original在图中只画均值水平线，其SD也要从原始metrics读取。

本包按原标签保留已有曲线。不能仅从“w/o Certainty”图例推断旧检索究竟用最终层还是全部层。新增行使用明确名称，不把旧标签自动改名成另一种检索配置。

旧目录结构应为：

```text
SOURCE_RUN/
  candidates/shared.jsonl.gz
  teacher/ours_all.jsonl.gz
  eval/original/metrics.json
  eval/ours/update-50/metrics.json
  ...
ABLATION_RUN/
  eval/no_certainty/update-50/metrics.json
  eval/no_similarity/update-50/metrics.json
  eval/random20/update-50/metrics.json
  ...
```

先导入，无GPU开销：

```bash
cd /root/INFLOW_mc_rebuttal_ablation
SOURCE_RUN=/你的旧主实验目录 \
ABLATION_RUN=/你的旧图3消融目录 \
ACTION=import bash scripts/run_rebuttal_mc_ablation.sh
```

检查输出 existing_results.csv。reused_raw_metrics 表示读到原始数值；missing_raw_metrics 表示该路径缺失，需指向正确旧目录或找回旧指标文件。导入操作不会重新评测或训练已有变体。

## 新增五行

| 名称 | 检索 | 初始/有检索采样 | 排序 |
|---|---|---|---|
| last_layer_certainty | 最终层相似度+来源certainty | 1/5 | JS |
| last_layer_no_certainty | 最终层相似度 | 1/5 | JS |
| sample_1_1_js | 原Ours检索集合 | 1/1 | JS |
| sample_5_5_js | 原Ours检索集合 | 5/5 | JS |
| sample_1_5_hard_flip | 原Ours检索集合 | 1/5 | argmax是否翻转 |

1/1直接取原5次有检索输出的第一条，沿用原无检索回答及分布，训练时使用这一条有检索回答。1/5 Hard flip完全复用原5个教师分布，先求它们的均值再取argmax。分数仅为0或1，用独立于JS的固定随机序列解决并列，再保留20%。

5/5补充5个无检索回答，每条得到合法选项上的完整概率分布，再求均值。它复用原5条有检索回答与原检索集合，从而只改变belief estimator。无检索批次的起始种子按原teacher入口的 seed + offset + pool_index 配对，默认offset为0（ours）。请核对旧运行实际的种子设置；不同前缀下，同批次种子不代表解码噪声完全消除。初始5条缓存在initial_five.jsonl.gz，中断后按题恢复。

最终层新检索行逐题核对原检索集合，集合与提示顺序相同的题直接复用原教师缓存，其余题才生成。新变体选出的有序训练输入、教师回答和demo顺序都与旧ours子集相同时，自动导入旧ours对应步数的metrics并跳过训练和评测，记录reuse.json。调用者需提供旧实验真实配置、权重和种子；仅凭截图标签不能判定协议一致。

## 启动

```bash
cd /root/INFLOW_mc_rebuttal_ablation
SOURCE_RUN=/你的旧Ministral主实验目录 \
ABLATION_RUN=/你的旧Ministral图3消融目录 \
MODEL=/你的Ministral权重目录 \
CONFIG=/你的旧实验配置.json \
DATA_FILE=/你的原选择题pool_test.jsonl.gz \
UPDATES=250 \
bash scripts/run_rebuttal_mc_ablation.sh
```

环境沿用原实验，依赖见requirements-autodl.txt。代码放系统盘，输出默认位于/root/autodl-tmp/experiments/ministral_mc_rebuttal_ablation。每个新变体用250×原batch_size条轨迹，调用原MC实验的opsd_train与opsd_evaluate入口，保留原500步调度并在250步停止，评测3次rollout。已有变体不会启动训练。

先运行最便宜的缓存重算对照，可以设置：

```bash
VARIANTS=sample_1_5_hard_flip,sample_1_1_js
```

VARIANTS等配置固定后按同一命令恢复；更改设置使用新OUTPUT。原训练入口支持完整模型缓存复用，不具备本包新增的逐update训练恢复，训练中断后该新变体会重训。教师/无检索回答准备按题恢复，已完成的新变体训练和评测不会重复。

输出existing_results.csv、all_results.csv和ablation_table.tex。旧行缺失时表格填--，不会从图像估算结果，也不会自动补跑旧行。
