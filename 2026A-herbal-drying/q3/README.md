# 第三问运行说明

正式运行入口为项目根目录的：

```powershell
python -B run_all.py --question 3 --validate
```

它读取 `validation/q3_selected_config.json` 中经验证的配置，读取第二问完整960单元末态，输出 `q3/solution.npz`、`submission_results/result3.xlsx`、表5、图5、精确终点记录与LaTeX正文。检查中重新从原始数组核算正式选档的必需加密比较。

`solve_q3.py` 是支持参数试算的底层求解器，其函数默认步长不承担正式选档功能；正式交付统一使用上面的入口。试验其他设置时指定单独的输出文件，保留已验收结果。

## 重做验证记录

在项目根目录执行下列命令。默认可复用指纹匹配的已保存算例，增加 `--force` 会重算对应矩阵。

```powershell
python -B q3/convergence_test.py --matrix time --time-steps 1 0.5 0.25 0.0625 0.03125 --workers 3
python -B q3/convergence_test.py --matrix space --space-step 0.25 --workers 3
python -B q3/convergence_test.py --matrix event --event-base q3/solution.npz
python -B q3/convergence_test.py --matrix event_step --event-base q3/solution.npz
python -B q3/convergence_test.py --matrix picard --event-step 1 --workers 2
python -B run_all.py --question 3 --validate
python -B validation/q3_reproduction.py
python -B validation/q3_report.py
```

时间矩阵保留粗档失败记录，整体标志可能为false；正式选档只依据 `q3_acceptance.json` 中重新核算的必需比较。事件复核从保存的同一完整锚点重新积分，改变局部步长或夹逼容差，不重复计算相同的前缀。空间矩阵的各个网格从各自的第二问完整末态开始，没有对960单元结果作网格投影。

## 最小复现

附录程序不导入主项目模块。全过程独立重算命令见 `paper_appendix_minimal/README.md`，主要入口为：

```powershell
python -B paper_appendix_minimal/minimal_example.py --question 3 --from-initial --dt-max 0.03125
```

这会独立重算第二问全部3小时并完成第三问。`validation/q3_reproduction.py` 只比较现有复现文件，不启动数值模拟。复现文件路径可通过其命令行参数指定。

## 查看结果

- `endpoint.json`：未舍入达标总时间、严格判据、夹逼区间；`native_grid` 为全部原网格单元，外层径向数组为重构采样。
- `tables/table5.md`：每6小时及精确终点的正文表；`tables/endpoint.csv`：精确终点对应21个规定位置的四位小数展示值。
- `figures/fig5_q3_drying_endpoint.pdf`：全域阈值交叉和含水率时空分布；同名SVG和PNG可用于编辑或预览。
- `figures/fig8_q3_convergence.pdf`：时间步和网格收敛补图。
- `../q4/figures/fig8_combined_convergence.pdf`：全文图8，汇总第三、四问空间与时间终点加密；插图LaTeX见 `../q4/数值验证配图.tex`。
- `论文正文_问题3.tex`：从正式结果自动生成的正文片段，需在主论文导言区加载中文、amsmath、graphicx与booktabs支持。

CSV展示值和Excel不作为续算输入；精确状态保存在 `solution.npz`。该结果基于固定尺寸、附录3物性及既定环境延续，不含第四问收缩。

`runs` 中正式选档的原始档位和被报告引用的粗档均是复核证据；`solution.npz` 与其选中运行副本还用于核对正式提升是否保持数组不变。空间比较另外需要 `../validation/runs` 中各自网格的第二问末态。独立复现核对使用 `../paper_appendix_minimal/reproduced/q3` 及 `q3_prefix` 的结果和来源记录。保留这些目录可直接复核现有证据，不能只发布Excel和正文图。

当前工程已实现第一至四问；本说明专述第三问。历史归档的“本地生成”仅描述当次归档操作，当前发布版本以Git历史和远程分支为准。
