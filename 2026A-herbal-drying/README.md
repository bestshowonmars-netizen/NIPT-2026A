# 2026 A题药材烘干：第一至四问

本项目已完整实现第一至四问。第一问计算0—1800 s，第二问计算0—10800 s，第三问承接第二问继续计算至全域达标。模型为一维径向有效传热传质、同心圆柱壳有限体积、全隐式时间推进及Picard迭代。第二问从题设均匀初态重新计算，统一采用附录3物性；第三问读取第二问完整末态，继续求解温度和含水率，内部物性模式仍为 `question=2`。

第四问从题设初态重新计算，使用附录4物性和附件2给定半径，在固定材料参考域内同步变换内部输运与表面交换。另有附录4固定半径对照，用于单独讨论本模型中收缩的作用。

## 先看哪些文件

- `submission_results/result1.xlsx`、`result2.xlsx`：题目要求的完整结果，两个工作表均为数值四位小数。
- `submission_results/result3.xlsx`：第三问单工作表结果，每60 s、每0.1 cm输出含水率。
- `submission_results/result4.xlsx`：第四问每60 s、固定物理位置每0.1 cm输出，域外留空，W列单独保存实际移动表面的含水率。
- `q4/tables/table6.md`、`q4/endpoint.json`：第四问正文表与未舍入终点；`q4/solution.npz` 和 `q4/fixed_control.npz`：收缩模型及同物性固定半径对照的完整结果。
- `q4/figures`：图6半径输入及坐标映射、图7移动区域含水率和三组模型比较、图8第三及第四问空间与时间终点加密；`q4/数值验证配图.tex` 提供总图8和第四问补图的插图片段。`validation/问题4精度验证报告.md`、`q4/论文正文_问题4.tex` 为验证记录与LaTeX正文。
- `q1/tables/题目规定表格.md`、`q2/tables/题目规定表格.md`、`q3/tables/题目规定表格.md`：表1—5；各表另有CSV。
- `q3/endpoint.json`：达标总时间、严格判据夹逼区间、最大值位置及完整网格终态。非整分钟终点不混入规定每分钟输出的Excel。
- `q3/方法草稿与结果解释.md`、`q3/续算与终点算法.md`：第三问的论文说明与算法细节。
- `q3/论文正文_问题3.tex`：由已核验结果生成的LaTeX正文片段，含方法、正文表、图5引用与验证差值。
- `validation/问题3精度验证报告.md`、`validation/q3_selected_config.json`：第三问单独的收敛记录及经验证选定的正式配置。
- `q1/方法草稿与结果解释.md`、`q2/方法草稿与结果解释.md`：交给论文手的短说明。
- `计算过程.md`：模型口径、离散公式、求解顺序、输出重构和水量核算。
- `validation/精度与复现报告.md`：最终空间/时间/Picard比较、完整结果核验及独立附录复现结果。
- `q1/figures`：配图方案的图1—3；`q2/figures`：图4；`q3/figures`：图5和第三问收敛补图；`validation/figures`：原第一、二问场变量收敛补图。每图均有PDF、SVG、300 dpi PNG。
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
python -B run_all.py --question 3
python -B run_all.py --question 3 --validate
python -B run_all.py --question 4 --validate
python -B run_all.py --recompute
python -B run_all.py --validate
python -B validation/result_checks.py --core
python -B validation/result_checks.py
python -B paper_appendix_minimal/minimal_example.py
```

默认运行复用已经保存且配置匹配的正式高精度数组，重新生成表格、图和解释。第一、二问修改求解器后应使用 `--recompute`；第三问缓存另检查源文件、输入和配置指纹，不匹配即重算。`--recompute` 强制重新求解，`--compute-only` 只计算，`--no-plots` 生成结果表但不画图。第一、二问 `--validate` 运行严格加密与检查；第三问 `--validate` 核对续算接口、完整数组和交付文件，独立加密矩阵通过 `q3/convergence_test.py` 运行，具体命令见第三问验证报告。`python -B validation/q3_report.py` 重生成该报告与配图。计算时长由CPU决定，多组加密与独立附录复现会明显长于单次求解。

## 结构与职责

`common` 包含附件读取、分段线性插值、物性、对流边界、有限体积内核、统一求解器、Excel导出和绘图。新增 `common/continuation.py` 复用原有限体积运算，提供完整状态续算、边界重构和全域最大值接口。`q3/solve_q3.py` 组织绝对时间续算，`q3/end_time_detector.py` 对交叉区间实际细步积分，`q3/analysis.py` 与 `q3/plotting.py` 从同一份高精度结果生成表图。`run_all.py` 是统一入口。

`data/attachment1.xlsx`、`data/attachment2.xlsx` 和四个 `data/templates` 文件是原始附件的逐字节副本。题目和原附件目录仅作为读取来源；所有生成内容均位于本项目。第三问保持固定半径0.02 m；第四问独立使用附录4与收缩输入，新增代码位于 `q4`，没有改变第一至三问的数值内核。

第四问的统一入口读取 `validation/q4_selected_config.json` 中独立加密选定的配置，同时生成固定半径对照。`--validate` 重新核对收敛证据、源文件指纹、水量、终点重积分和Excel逐格内容，不会自动重跑整个加密矩阵。完整重算、附录复现与检查命令见 `q4/README.md`。

目录保留 `data`、`common`、`q1`—`q4`、`validation`、`submission_results` 和 `paper_appendix_minimal` 的职责划分。各问的正式NPZ用于复用未舍入状态；`q3/runs`、`q4/runs`、`q4/cases` 以及 `validation/runs` 中被验证记录引用的NPZ是加密比较与来源核对证据。即使某档未通过或与正式结果数值相同，也可能仍由选档验收、图8或失败历史引用，不应只按“候选”名称删除。`paper_appendix_minimal/reproduced` 保存独立运行及逐格比较的证据。运行依赖缓存、临时预览和未被引用的试算副本不属于复核所需数据。

第三、四问交付说明同时记录各阶段的本地归档内容。归档生成器只制作压缩包，不执行远程发布；仓库是否已推送及对应版本以Git历史和远程分支为准。

## 数值输出约定

- 内部使用m、s、K；Excel和题目表按°C、cm、s或h输出。
- 第一、二问Excel从第1秒开始，第三问从第60秒开始；每0.1 cm一列。0秒初态保留在高精度数组和图中。
- `q1/solution.npz`、`q2/solution.npz` 保留未经舍入的101个径向采样值、每秒状态、计算网格、末状态和每秒诊断。题目要求的21个位置是其中每隔5列的子集。
- `q2/final_state.npz` 保存3小时完整网格状态、输入及配置，第三问直接读取，不能从绘图采样或Excel倒推该状态。
- `q3/solution.npz` 拼接前3小时的原有未舍入采样与后续结果，另保留全域最大值、精确终点、事件轨迹、累计失水及诊断。
- `q4/solution.npz` 同时保存固定物理位置和完整参考域采样、严格域内掩码、各时刻真实半径、单独表面值，以及完整初态、终点锚点和夹逼两端状态。计算体积、干物质权重固定在初始参考域，实际位置随半径移动。
- 第二问没有保存每秒的完整计算网格历史，因此第三问 `max_C`、`max_r` 在3小时前为NaN，图5最大值曲线从3小时开始；全程时空图仍包含前3小时的原有采样。没有把早期采样最大值当作全域计算网格最大值。
- 第三问的全域最大值包含全部有限体积单元、中心重构值和Robin表面重构值；严格判据使用未舍入数值。每6小时正文表与精确终点另存，非整分钟终点不加入Excel规则时间轴。
- 归一化水量单位为kg水/kg初始干物质，热物性的经验密度不用于重新构造干物质质量。
- 常规结果导出后可由Excel重新打开；不应把Excel四位小数倒读进求解器或当作内部高精度状态。

## 模型边界

环境水分指标被约定为药材侧的等效驱动，题给传质系数作配套的有效交换系数，不声称建立严格气固平衡。四问均不显式加入潜热，不纳入轴向及端部输运。第二至四问的表面对流系数约定沿用附录2。全部保持绝对时间，4小时后依既定模型假设延拓到50.0°C与0.0500 kg/kg。第四问进一步假定长度不变、径向同比例收缩，半径使用保形插值并在72小时以后保持附件末值。报告时长依赖这些物理与边界约定。

严格验收依据全部题目采样时空点的独立加密差异；这证明的是规定测试下的数值稳定性，不是对未知精确解的数学误差上界，也不代表完成内部温湿场实验验证。详细实测偏差见验证报告。
