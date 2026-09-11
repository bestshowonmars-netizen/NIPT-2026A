# 2026 A题药材烘干：第一、二问

本项目只实现第一问（0—1800 s）和第二问（0—10800 s）。模型为一维径向有效传热传质、同心圆柱壳有限体积、全隐式时间推进及Picard迭代。第二问从题设均匀初态重新计算，统一采用附录3物性。

## 先看哪些文件

- `submission_results/result1.xlsx`、`result2.xlsx`：题目要求的完整结果，两个工作表均为数值四位小数。
- `q1/tables/题目规定表格.md`、`q2/tables/题目规定表格.md`：表1—4；各表另有CSV。
- `q1/方法草稿与结果解释.md`、`q2/方法草稿与结果解释.md`：交给论文手的短说明。
- `计算过程.md`：模型口径、离散公式、求解顺序、输出重构和水量核算。
- `validation/精度与复现报告.md`：最终空间/时间/Picard比较、完整结果核验及独立附录复现结果。
- `q1/figures`：配图方案的图1—3；`q2/figures`：图4；`validation/figures`：场变量收敛补图。每图均有PDF、SVG、300 dpi PNG。
- `paper_appendix_minimal`：不导入主项目模块的附录最小闭环代码。

## 安装和运行

测试环境为 **64位 Python 3.12.14**。在项目根目录打开终端，安装 `requirements.txt` 中的公开Python包：

```powershell
python -m pip install -r requirements.txt
python -B run_all.py
```

建议将项目完整复制给队友。可以在自己的Python虚拟环境安装依赖；不需要MATLAB或Codex专用组件。本工作目录的 `.runtime/deps` 是已安装的项目内依赖，代码会优先使用它；在另一台机器无需复制这个目录，正常安装requirements即可。Numba首次运行要编译，之后复用缓存。`requirements.txt` 固定了实际运行版本。

常用命令：

```powershell
python -B run_all.py --question 1
python -B run_all.py --question 2
python -B run_all.py --recompute
python -B run_all.py --validate
python -B validation/result_checks.py --core
python -B validation/result_checks.py
python -B paper_appendix_minimal/minimal_example.py
```

默认运行复用已经保存且配置匹配的正式高精度数组，重新生成表格、图和解释。修改求解器源代码后必须使用`--recompute`重新计算；修改验证方法后用严格验证入口的`--recompute`。`--recompute` 重新求解；`--validate` 运行严格加密矩阵和检查，可以复用已完成的验证计算。要从头重做所有加密计算，使用 `python -B validation/strict_batch.py --recompute`。计算时长由CPU决定，严格加密与独立附录复现明显比单次正式计算耗时。

## 结构与职责

`common` 包含附件读取、分段线性插值、物性、对流边界、有限体积内核、统一求解器、Excel导出和绘图。`q1`、`q2` 只提供题目配置、入口、短分析及各自结果。`run_all.py` 是统一入口。`validation` 包含物理不变量检查、独立空间和时间加密、结果逐格核验及附录复现。

`data/attachment1.xlsx` 和两个 `data/templates` 文件是原始附件的逐字节副本。题目和原附件目录仅作为读取来源；所有生成内容均位于本项目。没有第三、四问代码、收缩计算或对应结果文件。

## 数值输出约定

- 内部使用m、s、K；Excel和题目表按°C、cm、s或h输出。
- 每份结果从第1秒开始，每0.1 cm一列；0秒初态保留在高精度数组和图中。
- `q1/solution.npz`、`q2/solution.npz` 保留未经舍入的101个径向采样值、每秒状态、计算网格、末状态和每秒诊断。题目要求的21个位置是其中每隔5列的子集。
- `q2/final_state.npz` 保存3小时完整网格状态、输入及配置，供以后继续推进；本次没有继续求干燥时长。
- 归一化水量单位为kg水/kg初始干物质，热物性的经验密度不用于重新构造干物质质量。
- 常规结果导出后可由Excel重新打开；不应把Excel四位小数倒读进求解器或当作内部高精度状态。

## 模型边界

环境水分指标被约定为药材侧的等效驱动，题给传质系数作配套的有效交换系数，不声称建立严格气固平衡。两问均不显式加入潜热，不纳入轴向及端部输运。第二问的表面对流系数约定沿用附录2。4小时后按指定规则延拓到50.0°C与0.0500 kg/kg，但本次两问均未使用区间外驱动。

严格验收依据全部题目采样时空点的独立加密差异；这证明的是规定测试下的数值稳定性，不是对未知精确解的数学误差上界，也不代表完成内部温湿场实验验证。详细实测偏差见验证报告。
