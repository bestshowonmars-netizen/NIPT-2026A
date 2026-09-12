# 口径对比与图件重绘

`build_v02_assets.py` 使用题目附件、正式第一至三问数组和材料坐标有限体积模型完成：

1. 问题三、四的环境末值延续与平均含水率判据对照；
2. 问题四 PCHIP/线性半径插值及附录 4 物性的固定半径对照；
3. 基准时间步连续减半至 1/8 的稳定性检查；
4. 论文 v02 的图 3、4、5、7及灵敏度图重绘。

模型沿用正文的一维径向有限体积、全隐式推进和 Picard 迭代。问题四仅使用材料坐标系数 `k/s²`、`D/s²`、`h/s`、`h_m/s`，未加入伪对流项。

结果见 `convention_comparison.csv` 和 `口径对比补算.json`。`data/` 是本地复核输入，不纳入 Overleaf 上传包。

运行时将项目主求解器的 `.runtime/deps` 加入 `PYTHONPATH`，然后执行：

```powershell
python -B -X utf8 recalculation/build_v02_assets.py
```
