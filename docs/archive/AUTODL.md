# AutoDL：四个模型共用一个实验入口

## 三任务主表和等时间预算对照

`bash scripts/run_autodl_open_table.sh` 默认先用 `/root/models/Qwen3-4B-Instruct-2507`。训练和测试各包含 MBPP 200、CoQA 150、SQuAD 150。Original 直接评测；Full、Redundancy、Consistency、NEURON、Mutual、INFLOW 每种训练 250 步，主表只做终点评测。每题采样三次，MBPP 用 pass@1，CoQA 与 SQuAD 用 EM，Overall 按题数加权。

`WALL_BUDGET_SECONDS` 默认 10800。正式运行前按独立计时预先确定预算，运行时固定并写入 `wall_clock_policy.json`。相同硬件下逐阶段计时，每个方法独立计入其所需的共享候选准备。预算包含模型加载、候选和教师生成、检索与特征、信念估计、选样、蒸馏与保存检查点。数据集构建和离线评测在预算外。

等时间对照在 250 步上限内选预算截止前保存成功的最晚检查点。每种方法至多多评一个预算端点；若选中 250 步则复用终点评测，预算内没有适配检查点则留空。`endpoint_table`、`cost_table`、`wall_clock_table` 均输出 CSV 和需要 `booktabs` 的 TeX 表，详见输出目录下的 `endpoint_costs.json` 和 `wall_clock_plan.json`。

新版代码放 `/root/INFLOW_paper_v3`。四个骨干共用一套 LLM 等价判断和预测熵加权的实验代码。通过 `MODEL` 切换权重路径，先运行 Qwen3-4B。

## 1. 上传代码到系统盘

上传 `INFLOW_figure2_paper_v3_autodl.zip` 后执行：

```bash
unzip -o /root/autodl-tmp/INFLOW_figure2_paper_v3_autodl.zip -d /root
cd /root/INFLOW_paper_v3
```

## 2. 沿用已有 PyTorch 2.5.1 镜像

```bash
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip install -r requirements-autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
PYTHONPATH=src python -m pytest tests/test_open_tasks_profiles.py tests/test_open_tasks_process.py -q
```

依赖列表不包含 PyTorch，无需重新下载。最后一条命令在服务器验证模型配置分派和 Linux 代码执行后端。

## 3. 先跑4B

确认系统盘上的4B目录含完整权重和 tokenizer。

```bash
tmux new -s inflow
cd /root/INFLOW_paper_v3
MODEL=/root/models/Qwen3-4B-Instruct-2507 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_autodl_open_500.sh
```

按 Ctrl+B 再按 D 离开；`tmux attach -t inflow` 返回。中断后用同一命令恢复。

## 4. 其余模型仍用同一个入口

逐个运行，只改 MODEL；以下为本地目录示例，须与实际下载目录一致：

```bash
MODEL=/root/autodl-tmp/models/Qwen3.5-9B bash scripts/run_autodl_open_500.sh
MODEL=/root/autodl-tmp/models/Ministral-3-8B bash scripts/run_autodl_open_500.sh
MODEL=/root/autodl-tmp/models/Llama-3.1-8B bash scripts/run_autodl_open_500.sh
```

4B权重放系统盘；9B及其余较大权重放数据盘。MODEL 必须指向已完整下载的本地目录。

## 5. 数据、训练和输出

- 共用数据：`/root/autodl-tmp/datasets/inflow_open_mixed_500_seed42`，TOPv2、MBPP各150题，CoQA、SQuAD各100题；训练和测试分别500题。
- 每个骨干评测 Original，并训练 Full、Redundancy、Consistency、NEURON、Mutual 和 InFlow（方法 ID 为 `js`）。Random 可通过 `METHODS` 显式追加，作为同检索的去 JS 消融。
- Figure 2 入口默认 500 次 optimizer update、batch size 4，每 50 步保留独立 adapter 并在同一批评测题上生成 3 次 rollout。250 步位于同一曲线上。只做 250 步终点评测可设置 `UPDATES=250 CURVE=0`。
- 输出路径为 `/root/autodl-tmp/experiments/inflow_<模型目录名>_seed42_paper_v3_curve`，包含 `checkpoints/<方法>/updates/step-000050` 等 adapter、`curve/step-000050` 等预测与评分、`figure2.pdf/.png/.csv` 和终点汇总。
- 多模型结果用 `PYTHONPATH=src python -m rig.open_tasks plot-curve --run <run1> <run2> <run3> <run4> --output <输出文件名>` 合并为 2×2 图。每个面板来自该骨干的真实检查点评测。
- 四个模型使用同一实验预算；架构相关 LoRA 目标自动适配。五个样本逐个生成以降低峰值显存。
- 设置 `SEED=43` 可切换种子；设置 `OUTPUT=/自定义路径` 可切换输出目录。改变实验配置时需使用新输出目录。

AutoDL配置使用 Linux 子进程执行 MBPP，含超时和资源限制；它不提供 Docker 同等的隔离。完整GPU训练的显存占用需在实际GPU上确认。本地验证未产生正式实验分数。

## 6. 复算旧预测

旧实验的 `results.csv` 只有聚合值。用旧 `predictions/*.jsonl`、相同 `eval.jsonl` 和旧 MBPP 执行结果可在独立目录重评分，区分 TOPv2 的结构、表面和严格 EM：

```bash
REVIEW="${OLD}_paper_v3_rescore"
mkdir -p "$REVIEW"
ln -s "$OLD/predictions" "$REVIEW/predictions"
export PYTHONPATH=src
for method in original full random js; do
  python -m rig.open_tasks score --config "$OLD/resolved_config.json" --data "$DATA_DIR" \
    --output "$REVIEW" --method "$method" --execution-results "$OLD/execution/$method/eval_results.json"
done
python -m rig.open_tasks report --output "$REVIEW"
```
