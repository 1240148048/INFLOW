# 开放式混合实验：500训练题 + 500测试题

**AutoDL普通容器用户请使用[AUTODL.md](AUTODL.md)和`scripts/run_autodl_open_500.sh`。该入口显式采用限资源Linux子进程，不要求嵌套Docker；下文通用入口默认仍使用Docker。**

四个数据集组成同一个训练池，每种比较方法训练一个跨任务模型。该实验独立于原选择题实验，不为每个任务各训练一次。

Qwen3-4B、Qwen3.5-9B、Ministral-3-8B、Llama-3.1-8B共用这一套代码和实验预算，切换本地MODEL路径即可。AutoDL入口共用500/500数据目录，各模型单独保存生成缓存、检查点和结果，并自动适配LoRA模块。

| 数据集 | 训练题 | 测试题 | 主要指标 |
|---|---:|---:|---|
| TOPv2 | 150 | 150 | 结构Exact Match |
| MBPP | 150 | 150 | 官方测试用例sampled pass@1 |
| CoQA | 100 | 100 | EM、token F1，按问答turn计数 |
| SQuAD v1 | 100 | 100 | EM、token F1 |
| **合计** | **500** | **500** | 按任务汇总，另报按题加权EM/pass@1 |

不使用HumanEval，不额外建立各任务500/500的实验。

## 一键运行

环境：Linux CUDA服务器、Python >=3.10、已安装匹配CUDA的PyTorch、Linux Docker。

```bash
cd INFLOW
python -m pip install -r requirements-open-tasks.txt
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/models/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=0 \
MODEL=/root/models/Qwen3-4B-Instruct-2507 \
OUTPUT=/experiments/inflow_open_mixed_500 \
bash scripts/run_open_mixed_500.sh
```

默认比较Original、Full、Random-20%、INFLOW-20%（命令名为js）。Original只评测，其余每种方法在混合池训练一次，均为250 updates。20%在混合500题上全局选择100题，不强制每个任务各保留20%。

脚本自动执行数据准备、500/500校验、五次无检索生成、跨层检索、五次有检索生成、选择、LoRA/EMA训练、分项评测，以及Markdown/CSV/TeX汇总。通用配置使用Docker执行代码；AutoDL配置显式选择限资源子进程，两者的隔离强度不同。

## 参数与恢复

| 环境变量 | 默认值/含义 |
|---|---|
| MODEL | 已完整下载的本地模型目录；自动识别四种骨干结构 |
| CONFIG | configs/open_mixed.json |
| OUTPUT | artifacts/open_mixed |
| DATA_DIR | OUTPUT/data，最终混合数据 |
| METHODS | full,redundancy,consistency,neuron,mutual,js；另支持random,hard_flip,entropy,agreement |
| SEED | 42 |
| UPDATES | 普通入口 250，AutoDL Figure 2 入口 500 |
| CURVE | 设为 1 时逐个评测已保存的 optimizer-update 检查点 |
| DATA_REVISION | HF数据revision，默认main |
| PYTHON | Python解释器 |
| EVAL_IMAGE | inflow-mbpp:py311 |

运行脚本自动设置PYTHONPATH，不必安装整个旧项目。独立依赖文件固定本次验证的Transformers/PEFT版本，不修改原模型流程。

混合入口题量固定为150+150+100+100。单任务调试入口为run_topv2_open.sh、run_mbpp_open.sh以及run_open_tasks.sh coqa或squad；它们不是本次主要实验。

数据、初始响应、teacher响应、预测均按题恢复。训练保存student/EMA adapters、optimizer、scheduler和数据遍历状态，可从最后checkpoint继续。配置/数据变化时拒绝复用旧结果，应使用新OUTPUT。

## 数据与评测协议

### TOPv2

- 使用WillHeld/top_v2的train/test划分，eval开发集不作为test。
- 按domain比例抽150/150；移除跨划分完全相同输入后再抽训练题，保持精确数量。
- 训练不保存每题gold parse；仅共享全局IN/SL标签schema，不提供gold示范。
- 主指标为结构 EM，比较意图标签、槽位标签、槽值和嵌套顺序，忽略意图节点中不属于任何槽位的抄写文本。槽值做 Unicode、大小写和切词归一化。表面 EM 比较整树并忽略切词差异，严格 EM 保留切词。三个指标分别报告。总体分为 TOPv2 结构 EM、QA EM 与 MBPP pass@1 的按题加权平均，各任务的百分点贡献单列。
- 辅助指标为root intent accuracy、层级slot F1和有效解析率。层级slot F1按祖先路径和完整slot子树匹配，是本实现明确给出的指标，不冒充其它官方slot协议。

### MBPP

- 从google-research-datasets/mbpp的full/train与full/test分别固定抽150题，保持官方边界。
- 训练只用题目。由 LLM 判断候选程序的逻辑等价性；官方测试仅用于独立评测。
- 测试prompt包含题目、函数名和不含expected outputs的输入调用表达式。通过测试AST提取接口，所有方法采用相同输入，不向模型提供断言答案。
- 官方test_list独立保存到eval_tests.jsonl，仅Docker评分读取；训练和模型生成阶段不读取。
- 每题生成3个候选，pass@1是每题成功比例的均值，不是至少成功一次的pass@3。
- 固定代码块提取规则，无效、超时计失败，不按方法单独修复代码。

### CoQA

- 使用stanfordnlp/coqa的train/validation；validation作为本实验held-out评测，不声称使用官方隐藏test。
- 100题指100个问答turn。固定seed打乱对话，按对话原顺序保留前缀直到100个turn，避免抽孤立中间问题。
- 训练准备的历史答案由初始模型生成；测试历史由当前被测模型生成，同一采样列沿用自己的前序回答，不填gold历史。
- 保留passage、历史问题及回答，实际测试多轮理解。
- 指标使用HF版本input_text的单参考EM/token F1，明确是模型生成历史协议，不标成官方多标注者CoQA排行榜F1。

### SQuAD v1

- 使用rajpurkar/squad的train/validation，各抽100题。
- 输入passage与question，输出简短答案；训练不保留gold答案或span。
- 评测小写、标点、冠词、空白归一化，对参考列表取最佳EM/token F1。

来源：[TOPv2](https://aclanthology.org/2020.emnlp-main.413/)、[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)、[CoQA](https://huggingface.co/datasets/stanfordnlp/coqa)、[SQuAD](https://huggingface.co/datasets/rajpurkar/squad)。

## INFLOW任务适配

1. 检索前后各生成5次。冻结的同一底座模型对回答和已有类代表做语义或程序逻辑等价判断。检索后的回答沿用检索前的类别，并可建立新类别。
2. 对每条回答，在实际生成条件与采样前缀下计算每个回答 token 的全词表预测熵，取长度均值。若采样到 EOS，则计入均值。
3. 每条回答按 `exp(-平均预测熵)` 加权，类内求和后归一化，再计算检索前后分布的 JS。
4. 无效输出在两侧共享一个类别。certainty 为有效概率质量乘以 `1-H(P0)/log(M)`。
5. 从无检索的语义分布中取概率质量最大的有效类，选择类内平均预测熵最低的示范。用这条示范的最后一个非 EOS/pad token 提取各层表示，以固定 Rademacher 投影到 64 维后逐层归一化。
6. InFlow 检索分数为 `-0.5*log(1-c_source*cos_positive²)`。为保持开放任务的输入输出契约，各种检索均限制在目标任务内，检索 3 条并排除自身。
7. JS 全局选 20%，Random 复用 InFlow 检索并随机选 20%。Full 用末层余弦并保留所有题。Redundancy 共用 Full 生成缓存，迭代移除末层表示最冗余的题。Consistency 无检索，用初始多数类产生自反馈，再按五次回答的模态类比例筛选。NEURON 用正贡献神经元 Jaccard 检索，保留平均活跃集合最小的题。Mutual 以层表示估计高斯条件互信息，贪心构建检索集并按累积条件增益选题。除 Full 外，均保留 20%。
8. 每步重新从当前student采样；student不带检索、EMA teacher带检索，在相同student prefixes上计算token-mean reverse KL。固定采样前缀后仅对student分布反传。
9. rank-16 LoRA，batch size 4，EMA retention 0.995。Figure 2 运行至 500 步，按 50 步保存和评测。每个数据点取三次 rollout-index Overall 准确率均值，阴影为这三列的总体标准差。
10. 混合配置允许12,288 prompt tokens、16,384 teacher序列tokens以容纳多轮历史。超长直接报错，不静默截掉问题/历史。正式实验前可按显存统一调整CONFIG，之后固定。

## 输出

```text
OUTPUT/
  components/{topv2,code,coqa,squad}/
  data/{train,eval,eval_tests}.jsonl
  data/data_manifest.json
  resolved_config.json
  experiment.json
  contracts.jsonl
  initial.jsonl
  retrieval.json
  teacher.jsonl
  preparation_metrics.json
  subsets/{full,random,js}.jsonl
  checkpoints/METHOD/student/
  checkpoints/METHOD/training_state.pt
  predictions/METHOD.jsonl
  predictions/METHOD.code.jsonl
  execution/METHOD/eval_results.json
  metrics/METHOD.json
  results.md
  results.csv
  results.tex
  paired_comparisons.json
```

每方法一个跨任务模型。results按四任务和Overall分行；Overall为按500题加权的EM/pass@1，不混入token F1。公共预处理时间单列，不能用training_seconds代替总成本。

普通配对bootstrap以题为单位，保留同题候选整体。CoQA的turn存在对话内相关性，须同时考虑对话成组区间；普通逐题区间不代表独立对话或训练种子变异。

## 分阶段执行

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
for task in topv2 code coqa squad; do
  python -m rig.open_tasks data --task "$task" --output "artifacts/components/$task"
done
python -m rig.open_tasks mix --topv2 artifacts/components/topv2 --code artifacts/components/code --coqa artifacts/components/coqa --squad artifacts/components/squad --output artifacts/open_data
python -m rig.open_tasks prepare --config configs/open_mixed.json --data artifacts/open_data --output artifacts/open_run
python -m rig.open_tasks select --config configs/open_mixed.json --output artifacts/open_run --methods full,random,js
python -m rig.open_tasks train --config configs/open_mixed.json --data artifacts/open_data --output artifacts/open_run --method js
python -m rig.open_tasks predict --config configs/open_mixed.json --data artifacts/open_data --output artifacts/open_run --method js
bash scripts/eval_mbpp_docker.sh artifacts/open_run/predictions/js.code.jsonl artifacts/open_data/eval_tests.jsonl artifacts/open_run/execution/js
python -m rig.open_tasks score --config configs/open_mixed.json --data artifacts/open_data --output artifacts/open_run --method js --execution-results artifacts/open_run/execution/js/eval_results.json
python -m rig.open_tasks report --output artifacts/open_run
```

## 验证

```bash
PYTHONPATH=src python -m pytest tests/test_open_tasks.py tests/test_open_tasks_training.py tests/test_opsd_train.py -q
```

测试覆盖标签隔离、跨层检索、自身排除、默认规模、多轮自身历史、混合分项评分、pass@1、缓存/训练恢复，以及真实小型causal LM+PEFT上的重新采样、共享前缀和teacher冻结梯度。

正式结果需要CUDA训练和Docker执行。CPU测试及数据加载不作为论文结果，结果表仅读取真实评测文件。
