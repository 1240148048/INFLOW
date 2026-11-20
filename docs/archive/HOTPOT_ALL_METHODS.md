# 开放式三任务：共享初始回答 1/5 协议

## 启动

上传 `INFLOW_hotpot_joint15_js_only_cluster.zip` 至 `/root/autodl-tmp/`：

```bash
unzip -oq /root/autodl-tmp/INFLOW_hotpot_joint15_js_only_cluster.zip -d /root
cd /root/INFLOW_open_mixed_500
bash scripts/run_autodl_hotpot_all.sh
```

默认 Qwen3-4B 位于 `/root/models/Qwen3-4B-Instruct-2507`。可用 `MODEL=/模型目录 bash scripts/run_autodl_hotpot_all.sh` 更换模型，四种模型共用同一套代码。

若之前通过官方原始 JSON 解决下载问题，继续使用：

```bash
bash scripts/run_autodl_hotpot_direct.sh
```

如果本地已有 `/root/autodl-tmp/datasets/raw_hotpot/hotpot_train_v1.1.json`，该入口直接复用它。

## 数据与训练

每个划分包含 HotpotQA medium 200、CoQA 150、SQuAD 150，共 500 题。HotpotQA 从官方 train-medium 抽取 400 道互不重叠题，分为 200 训练和 200 留出测试。旧的已准备数据可直接复用。

运行 Original、Full、Redundancy、Consistency、NEURON、Mutual、JS。Original 只评测，其他方法分别训练 250 optimizer updates，仅测最后节点，每题采样 3 次。所有任务直接输出最终答案，不输出中间推理、解释或其他内容，关闭 thinking。温度保持 0.6，top_p 保持 0.95；初始回答和检索后回答采用相同采样参数。

## 初始回答、检索与选择

Full、Redundancy、NEURON、Mutual 和 JS 共用 `initial.jsonl`：每题只生成 1 个初始回答，保存该回答的所有层末 token 表示与 full-vocabulary predictive entropy 的 token 均值 H。源 certainty 为 `exp(-H)`。不再由多次回答的语义分布计算 certainty，也不再从五个回答中挑一个初始代表。

- JS：基于单个初始回答的全层轨迹 similarity 与源 certainty 计算 Gaussian information 检索分数。
- Full、Redundancy：保留选择题实验的 final-layer cosine 检索设计。
- NEURON：保留初始回答贡献神经元集合的 Jaccard 检索与检索后平均集合大小的选择规则。
- Mutual：对齐选择题实现，使用初始回答全层 similarity 和源 certainty 计算各边 Gaussian information；选样分数采用所选边的合并信息量。移除之前额外引入的条件信息贪心检索。
- Consistency：不生成、不读取共享初始回答，不检索，也没有额外 self-check 草稿提示。独立生成 5 个回答，按规范化答案的最大频次除以 5 排序，不调用 LLM 做语义聚类。CoQA 使用 Consistency 自己的历史回答。

三种 QA 的输出契约相同，因此在 500 题池内检索，不按数据集分池。表示保留选择题代码中的既有随机投影设置，默认每层 64 维。

Full 用完整训练池；其他选择方法取 20%。JS 在全池按分数统一排序取 100 题，不设置任务配额。Redundancy 与 Full 共用对应 teacher 缓存，其余 family 保留各自 teacher。

## 联合语义聚类和 JS

仅对 JS，每题将初始回答 1 个与检索后回答 5 个同时输入一次 LLM，请它返回六个回答的语义等价划分。类别编号按最早出现的位置规范化，初始回答所在类别是 class:0。不同实体、否定、数值和实质限定条件应分开；等价缩写和改写可合并。判断等价而非正确性，不把相关但错误的回答标成 invalid。只有实际空文本使用空答案类别。

初始分布 P0 是初始类别的 one-hot。检索后第 m 个回答权重为 `exp(-H_m) / sum_j exp(-H_j)`，其中 H_m 是该回答的 length-normalized predictive entropy。将五个权重按联合类别累加形成 PS，再计算 JS(P0, PS)。不使用 token NLL 替代预测熵，不增加任务配额，不提高采样温度。

Full、Redundancy、NEURON、Mutual、Consistency 均不调用 LLM 聚类，也不计算语义分布或 JS。需要选择代表回答或统计一致性时，仅对答案做固定的大小写、标点、冠词、空白规范化并直接计票；不让 LLM 合并别名或改写。Consistency 不混入共享初始回答。

## 文件

- `initial.jsonl`：共享的单次初始回答、预测熵、certainty 和检索特征。
- `teacher.jsonl`：JS 的 `initial_response`、`retrieved_responses`（5个）、`joint_responses`（6个）、`joint_class_ids`、`joint_clusters`、两侧分布和分数。
- `teacher.full.jsonl`、`teacher.neuron.jsonl`、`teacher.mutual.jsonl`：对应 baseline 的 1/5 记录，不包含语义聚类结果或 JS 分数。
- `teacher.consistency.jsonl`：独立 `independent_responses`（5个），`initial_response` 为 null。
- `subsets/js.jsonl`：同时保存明确命名的前后回答及分布，移除歧义的顶层 `responses` 字段。
- `results.csv`、`results.tex`、`results.md`：全部方法的分任务及 Overall 结果。

默认实验目录：

```text
/root/autodl-tmp/experiments/inflow_Qwen3-4B-Instruct-2507_seed42_hotpot_medium_all_250_joint15_jscluster_v8/
```

该目录与之前 5/5 及所有方法都聚类的版本分开；不能复用旧 initial、teacher、选样或训练 checkpoint。原数据划分可复用。相同新版命令可恢复新版实验，逐题打印准备、每次 update 和逐题评测进度。保存的学生 adapter、EMA teacher、优化器、调度器与数据顺序用于训练恢复。
