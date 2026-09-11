# 第四问：给定半径历程下的干燥计算

第四问从 \(t=0\)、\(T=301.15\,\mathrm K\)、\(C=2.55\,\mathrm{kg/kg}\) 重新求解。不会读取问题2或问题3末态。附录4物性、固定参考网格、全隐式时间推进、Picard迭代和全域事件定位构成统一计算流程。

\[
r=s(t)x,\quad s(t)=\frac{R(t)}{R_0},\qquad
\widehat k=\frac{k_4}{s^2},\quad \widehat D=\frac{D_4}{s^2},\quad
\widehat h=\frac{h}{s},\quad \widehat h_m=\frac{h_m}{s}.
\]

参考网格体积与干物质权重保持不变。每步用与输运相同的隐式表面通量累计失水，检查

\[
B_4(t)=\sum_i\frac{V_i^0}{\sum_j V_j^0}C_i(t)+L_4(t)-2.55.
\]

`shrinking_radius.py` 读取原半径数据并执行PCHIP；`kernel.py` 在参考域推进与重构；`solve_q4.py` 组织从初态至全域达标的状态保存；`fixed_radius_control.py` 提供附录4固定半径对照入口；`end_time_detector.py` 从未达标锚点实际重积分候选时刻；`analysis.py` 和 `plotting.py` 统一生成表图。

## 运行与验证

在工程根目录运行：

```powershell
python -B run_all.py --question 4 --validate
python -B run_all.py --question 4 --recompute --validate
python -B validation/q4_checks.py --core
python -B validation/q4_acceptance.py
python -B validation/q4_report.py
python -B validation/q4_reproduction.py --run
```

统一入口使用 `validation/q4_selected_config.json` 中经过第四问独立验证的正式设置。已保存的结果只在输入、数值代码和配置指纹匹配时复用；`--recompute` 将从题设初态重新求解第四问及附录4固定半径对照。

需要试验其它配置时，可直接运行求解入口并指定独立输出文件，避免覆盖正式结果：

```powershell
python -B q4/solve_q4.py --dt 1 --output q4/cases/trial.npz
python -B q4/solve_q4.py --dt 1 --fixed-radius --output q4/cases/fixed_trial.npz
```

上述 `--dt 1` 是试验设置，并非正式精度承诺。具体加密档次和实测差见 `validation/问题4精度验证报告.md`。网格加密的每个算例均从初态重算，不投影其它网格的末态。

## 输出口径

- `solution.npz`：正式收缩模型；`fixed_control.npz`：附录4固定初始半径对照。两者内部都计算温度和含水率。
- `times` 保留0秒、每60秒及精确终点。`moisture`、`temperature_K` 对应固定物理位置 `radii`；`valid_mask=False` 的域外位置是NaN。
- `moisture_ref`、`temperature_ref_K` 对应完整材料参考位置 `reference_radii`，实际位置为 `xi*radius_m`。绘制移动区域热力图应使用这套完整参考场。
- `surface_moisture` 单独对应当前 \(r=R(t)\)，由Robin条件重构。不能用固定2cm列或最外层单元值代替。
- 全域最大值来自全部计算单元、重构中心及实际表面。`endpoint.json` 保留严格达标总时长、夹逼符号、未舍入半径、完整参考/物理网格与终点状态。
- `submission_results/result4.xlsx` 从60秒开始，以秒为时间单位，每0.1cm一列；域外留空，W列为“药材表面”。非整分钟终点另存于正文表6与终点记录。
- `tables/table6.md`、CSV按每6小时和精确终点整理。表中0.1500是四位小数显示，达标判断只使用未舍入数值。

图6、图7及汇总第三、四问收敛记录的图8均提供PDF、SVG及300dpi PNG。图7同时展示第三问、附录4固定尺寸、第四问收缩模型；只有后两组之差用于讨论同物性条件下收缩的作用。数值稳定性检查与独立程序复现均不等同于内部场实验验证。

全文图8为 `figures/fig8_combined_convergence`，其插图代码与第四问独立收敛补图见 `数值验证配图.tex`。图8使用已有第三、四问的参数匹配序列，参考点是各序列的最细离散结果，不能将点间差异解释为未知真解误差上界。

`runs`、`cases` 中被 `../validation/q4_selected_config.json`、加密报告及图8引用的NPZ与配套JSON用于重用和复核。正式 `solution.npz`、`fixed_control.npz` 之外仍需保留对应选中运行副本；未通过的960单元空间比较用于说明最终选择1280单元的依据。`../paper_appendix_minimal/reproduced/q4` 保存独立复现输出和Excel，供 `q4_reproduction.py` 逐值核对。临时预览、运行缓存和未被引用的试算不属于正式证据。

当前工程已完整实现第一至四问。本地交付归档不会自行推送GitHub；发布状态及对应版本以Git历史和远程分支为准。
