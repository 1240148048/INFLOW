# JS 单方法入口

当前实现采用共享单次初始回答、检索后 5 次回答、六个回答联合语义聚类。详细方法、公式和全部 baseline 设置见 [HOTPOT_ALL_METHODS.md](HOTPOT_ALL_METHODS.md)。

只运行 JS：

```bash
bash scripts/run_autodl_js_task_prompt.sh
```

它不会自动运行 Original。全池排序选 20%，没有任务配额。温度 0.6、top_p 0.95，250 updates 仅测最终节点。

全部方法运行：

```bash
bash scripts/run_autodl_hotpot_all.sh
```
