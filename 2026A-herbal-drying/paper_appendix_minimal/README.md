# 论文附录最小闭环代码

第四问入口为 `minimal_q4.py`，从题设初态读取附件1、附件2和result4模板，联合计算温度与含水率至全域严格达标。它复用本目录的 `minimal_solver.py` 中通用有限体积运算，自包含附录4物性、PCHIP、材料坐标系数变换、实际域采样和事件定位，不导入主项目模块。

```powershell
python -B paper_appendix_minimal/minimal_example.py --question 4
python -B validation/q4_reproduction.py
```

第四问默认数值设置由经验证的正式选档固化到程序中，输出到 `paper_appendix_minimal/reproduced/q4`。结果文件包含完整计算网格终态、每分钟固定物理位置、独立移动表面、域外掩码和水量核查；`result4.xlsx` 域外留空，非整分钟终点另存。复现检查逐项对比未舍入数据和Excel全部数值及空格。

可独立拷贝 `minimal_solver.py`、`minimal_q4.py` 及两个原附件、result4模板，运行：

```powershell
python -B minimal_q4.py --attachment attachment1.xlsx --radius attachment2.xlsx --template result4_template.xlsx --output reproduced
```

增加 `--fixed-radius` 即运行附录4固定尺寸对照。这里“独立复现”是同一数值方法的独立运行，不表示更换物理模型或完成实验验证。

`minimal_solver.py` 自包含环境读取、附录2/3物性、边界条件、径向有限体积、全隐式Picard、中心/表面重构和模板导出。`minimal_q3.py` 在同目录内核上完成第三问长时间续算与真实积分的终点定位。它们不导入主项目的common、q1、q2、q3或run_all，只依赖Python标准库、NumPy、Numba和openpyxl。`minimal_example.py` 是统一入口包装。

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

## 第三问从初态到达标的闭环

在项目根目录运行：

```powershell
python -B paper_appendix_minimal/minimal_example.py --question 3 --from-initial --dt-max 0.03125
```

该命令先由附录内核从附件1重新计算第二问全部3小时，再按绝对时间推进第三问，输出 `reproduced/q3/q3_independent.npz`、`result3.xlsx` 和 `endpoint.json`。网格为960单元，第二问时间尺度为1/768 s，第三问正式最大步长为1/32 s，事件步长不超过普通步长，夹逼容差为0.01 s。该档已经通过本次预定的加密比较，具体差值见主项目第三问验证报告和选档文件。

若已经独立重算了第二问，可以避免重复：

```powershell
python -B paper_appendix_minimal/minimal_example.py --question 3 --prefix paper_appendix_minimal/reproduced/q3_prefix/q2_independent.npz --dt-max 0.03125
```

不带 `--from-initial` 或 `--prefix` 时，默认读取主项目 `q2/solution.npz` 的完整末态；这种运行仅复现第三问续算，不能称为从原始数据独立重算全过程。证据中记录实际前缀路径、SHA256和附件SHA256。默认示例仍运行第一、二问，第三问需显式选 `--question 3`。

复算完成后，在项目根目录执行 `python -B validation/q3_reproduction.py`，逐数组、逐单元格核对默认 `reproduced/q3` 中的本次结果。已提供的全程复现证据在 `validation/q3_reproduction.json`；正式独立复现输出位于 `reproduced/q3`，字节相同的候选副本仅在本地保留。

独立拷贝第三问闭环所需文件为 `minimal_solver.py`、`minimal_q3.py`、`minimal_example.py`、原始 `attachment1.xlsx` 和 `result3_template.xlsx`。可在任意目录显式给定输入：

```powershell
python -B minimal_q3.py --from-initial --attachment attachment1.xlsx --template result3_template.xlsx --output reproduced --dt-max 0.03125
```

这里的“独立”指不导入主项目、独立从输入重算并核查交付，不指使用不同物理模型或不同数值方法。核心函数由 `validation/build_q3_appendix.py` 生成，以保持附录与主程序一致；运行附录不需要该生成脚本，也不需要原主项目模块。终点使用全网格和边界重构值的严格判据，结果文件仅保留规则60 s行，非整分钟终点单独保存。
