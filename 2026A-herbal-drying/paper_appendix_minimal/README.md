# 论文附录最小闭环代码

`minimal_solver.py` 自包含环境读取、附录2/3物性、边界条件、径向有限体积、全隐式Picard、中心/表面重构和模板导出。它不导入主项目的common、q1、q2或run_all；只依赖Python标准库、NumPy、Numba和openpyxl。`minimal_example.py` 是入口包装。

在Python 3.12（64位）环境安装主项目requirements中的包后：

```powershell
python -B minimal_example.py
```

默认读取上一级data中的附件1及两个模板，按与正式计算相同的960单元、时间尺度1/768秒和严格Picard容差计算两问。结果写入本目录的reproduced，不覆盖正式submission_results。

独立拷贝到另一目录时，可以显式指定输入：

```powershell
python -B minimal_solver.py --attachment attachment1.xlsx --templates templates --output reproduced
```

templates中应包含result1_template.xlsx、result2_template.xlsx。两问的结果包含各自完整规定时间范围，不是只跑几十秒的演示。仅需第一问时用`--question 1`，第二问用`--question 2`。

用于核验的重新计算高精度数组存为q1_independent.npz、q2_independent.npz。与主项目输出的全时空数值及最终Excel逐项比较，核验结果在主项目validation/精度与复现报告.md中。本代码保留必要的结果与诊断保存功能，适合连同输入说明放入论文附录；是否删去打印日志应由论文手自行决定，不影响数值方法。
