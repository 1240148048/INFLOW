# 从已有 subsets 补充五组消融折线图

本入口直接读取已有 `subsets/*.jsonl.gz`，不生成候选、不生成教师回答、不重新选样。
每组从同一原始模型训练到 500 optimizer updates，沿用原 500 步学习率调度，
在 50、100、150、200、250、300、350、400、450、500 共十个 checkpoint 上评测。
使用原配置的完整测试集，每题三次 rollout；均值为 Avg@3，SD 为 rollout-index SD，
不是多个训练 seed 的标准差。

在已有服务器代码目录解压补丁后运行：

```bash
bash scripts/start_rebuttal_mc_curve.sh
```

默认读取 `/root/autodl-tmp/experiments/ministral_mc_rebuttal_ablation/subsets`，
从同级 `experiment.json` 读取原模型、配置、数据及 candidates 路径。
输出到同级 `curve_500`，不覆盖旧的 250 步 adapter、评测和 CSV。
旧入口只保存 250 步终点，不能恢复早期 checkpoint，也不含优化器/EMA 状态；
因此新曲线从原始模型重训，但完全复用已经选好的 subsets。

新增训练仅包含：

- `last_layer_certainty`
- `last_layer_no_certainty`
- `sample_1_1_js`
- `sample_5_5_js`
- `sample_1_5_hard_flip`

旧四条对照曲线和 Original 从旧 `all_results.csv` 导入，不重训。
新曲线的所有点只读取新生成的逐 checkpoint `metrics.json`；
不会把旧 250 步单点混入新训练或插值补足缺失点。
旧来源 JSON 中的 digitized-estimate 标记保留在新来源 JSON 中。

同一命令重跑会复用已完成训练和已完成 checkpoint 评测。
训练中断时尚无优化器恢复功能，该组会从原始模型重训；生成和选样均不重做。
不要并发运行两个命令写入同一个 `curve_500`。

输出：

```text
curve_500/
  original_config.json
  curve_experiment.json
  checkpoints/<variant>/checkpoint-200/  # batch size 4 时为 update 50
  ...
  checkpoints/<variant>/checkpoint-2000/
  eval/<variant>/update-50/metrics.json
  ...
  eval/<variant>/update-500/metrics.json
  all_results.csv
  all_results_sources.json
  curve_progress.json
  supplementary_ablation_curves.png
  supplementary_ablation_curves.pdf
  supplementary_ablation_curves.svg
```

图按组件、检索层、belief/选样设置分为三个面板，阴影表示原评测定义的 SD。
每评完一个 checkpoint 就刷新 CSV 和进度；全部评完后自动绘图。
只重新绘图用 `ACTION=plot bash scripts/start_rebuttal_mc_curve.sh`；
只检查已有文件和配置、不启动训练用 `ACTION=plan bash scripts/start_rebuttal_mc_curve.sh`。

路径变更时可设置 `PREVIOUS_RUN`、`SOURCE_RUN`、`SUBSETS_DIR`、`MODEL`、`DATA_FILE`；
默认沿用旧配置快照，不需要重新安装依赖或下载模型。
