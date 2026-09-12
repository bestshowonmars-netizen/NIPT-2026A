"""Apply the reviewed v02 text and layout changes to main.tex once."""
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "main.tex"
text = path.read_text(encoding="utf-8")


def required(old: str, new: str) -> None:
    global text
    if old not in text:
        raise RuntimeError(f"expected source block missing: {old[:80]!r}")
    text = text.replace(old, new, 1)


def section(start: str, end: str, new: str) -> None:
    global text
    i, j = text.index(start), text.index(end, text.index(start))
    text = text[:i] + new + text[j:]


required("\\usepackage{booktabs,tabularx}\n", "\\usepackage{booktabs,tabularx}\n\\usepackage{tikz}\n\\usetikzlibrary{arrows.meta,positioning}\n")
text = text.replace("环境等效水分驱动", "环境水分指标")
required(
    "针对问题三，以全部网格单元及中心、表面重构值的最大值为严格判据，通过保存未达标状态、重新积分和二分区间约束定位首次达标时刻，得到总烘干时间约57.47~h（206900~s）。",
    "针对问题三，以全部网格单元及中心、表面重构值的最大值为严格判据，通过保存未达标状态、重新积分和区间缩小定位首次达标时刻，得到总烘干时间约57.47~h。快速口径对照表明，若改用平均含水率判据，终点将提前约21.4~h。",
)
required(
    "空间、时间、事件定位及Picard容差收紧检验均满足精度要求，四问最大水量收支残差不超过$2.54\\times10^{-12}$~kg/kg。",
    "网格和步长加密后，问题三、四终点变化均小于0.1~s；四问最大水量收支残差不超过$2.54\\times10^{-12}$~kg/kg。",
)
required(
    "题给水分数据按药材侧等效平衡驱动解释，不把空气含湿量与固体干基含水率的质量基准直接等同。",
    "题给环境水分指标作为药材侧的等效边界驱动，不把空气含湿量与固体干基含水率的质量基准直接等同。",
)
required(
    "模型不显式计入蒸发潜热、辐射和化学反应热，其影响纳入有效物性和边界系数的不确定性。",
    "模型不显式计入蒸发潜热、辐射和化学反应热，因此温度结果只在题给有效物性和边界参数的口径下解释。",
)
required(
    "$M(t)$ & 全域含水率最大值 & kg/kg\\\\\n$L(t),B(t)$ & 累计失水及水量收支残差 & kg/kg\\\\\n$N,\\Delta t,\\varepsilon_t$ & 控制体数、时间步长和事件定位容差 & 1，s，s\\\\",
    "$M_3(t),M_4(t),g(t)$ & 问题三、四的全域含水率最大值及阈值事件函数 & kg/kg\\\\\n$\\rho_d,L(t),B(t)$ & 干物质体积密度、累计失水及水量收支残差 & kg/m$^3$，kg/kg，kg/kg\\\\\n$S_p,\\tau$ & 归一化灵敏度和时间步长尺度 & 1，s\\\\\n$N,\\Delta t,\\varepsilon_t$ & 控制体数、实际时间步长和事件定位容差 & 1，s，s\\\\",
)

section(
    "% TODO（总体技术路线图）",
    "\\section{问题一的建模与求解}",
    r"""\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[
    node distance=5mm,
    box/.style={draw=black!65,rounded corners=2pt,fill=blue!5,minimum height=15mm,
                text width=3.0cm,align=center,font=\small},
    arrow/.style={-{Latex[length=2.2mm]},thick,draw=black!65}
  ]
    \node[box] (q1) {问题一\\常热物性传热\\非线性水分扩散};
    \node[box,right=of q1] (q2) {问题二\\变物性耦合\\重新从初态计算};
    \node[box,right=of q2] (q3) {问题三\\全域达标判定\\终止事件定位};
    \node[box,right=of q3] (q4) {问题四\\材料坐标变换\\给定半径收缩};
    \draw[arrow] (q1) -- (q2); \draw[arrow] (q2) -- (q3); \draw[arrow] (q3) -- (q4);
    \node[draw=black!50,fill=gray!7,below=7mm of q2,minimum height=9mm,
          text width=9.0cm,align=center,font=\small] (method)
          {统一数值框架：圆柱壳有限体积 $+$ 后向Euler $+$ Picard迭代\\
           统一输出：径向场、全域判据、水量守恒与网格/步长加密检验};
    \draw[arrow] (method.west) -| (q1.south);
    \draw[arrow] (method.east) -| (q4.south);
  \end{tikzpicture}
  \caption{四个问题的递进关系与统一求解框架}
  \label{fig:workflow}
\end{figure}

""",
)

for unit in ("s", "h"):
    old = f"$t$/{unit}&$r=0$&$r=0.5$&$r=1.0$&$r=1.5$&$r=2.0$ cm\\\\\\midrule"
    new = f"$t$/{unit}&\\multicolumn{{5}}{{c}}{{径向位置 $r$/cm}}\\\\\\cmidrule(lr){{2-6}}\n&0&0.5&1.0&1.5&2.0\\\\\\midrule"
    text = text.replace(old, new)
required(
    "$t$/h&$r=0$&$r=0.5$&$r=1.0$&$r=1.5$&$r=2.0$ cm&表面&$R(t)$/cm\\\\\\midrule",
    "$t$/h&\\multicolumn{5}{c}{固定物理位置 $r$/cm}&表面&$R(t)$/cm\\\\\\cmidrule(lr){2-6}\n&0&0.5&1.0&1.5&2.0&&\\\\\\midrule",
)

required(
    "图~\\ref{fig:q3endpoint}上图给出全域最大含水率及阈值附近的事件试探点，下图给出全过程含水率场和0.15等值线。中心最迟越过阈值，因此决定最终烘干时间。",
    "图~\\ref{fig:q3endpoint}从题设初态起统一展示全域最大值、径向平均值及三个代表位置的含水率变化。中心始终是最慢干燥位置，因而控制最终烘干时间；平均值则会更早越过阈值。",
)
required(
    "该方法不在两个输出值之间直接线性插值，因而候选时刻始终对应实际数值积分状态；最终事件定位误差小于0.01~s，并取首次满足严格判据的一侧作为终点。",
    "该方法不在两个输出值之间直接线性插值，因而候选时刻始终对应实际数值积分状态；最终搜索区间宽度小于0.01~s，并取首次满足严格判据的一侧作为报告终点。该区间宽度仅反映事件搜索精度，不代表模型预测误差。",
)
required(
    "设单位轴长内干物质质量守恒，则$\\rho_d(t)R(t)^2=\\rho_{d,0}R_0^2$，故$\\rho_d\\propto s^{-2}$。",
    "设单位轴长内各材料壳的干物质质量守恒，则$\\rho_d(t)R(t)^2=\\rho_{d,0}R_0^2$，故$\\rho_d\\propto s^{-2}$。这里$\\rho_d$只用于水量核算；热方程中的$a(C)=\\rho(C)c_p(C)$是题给经验式构造的有效热储存系数，二者不强加额外密度关系。",
)
required(
    "终点时未舍入的全域最大含水率刚好低于0.15~kg/kg，位于中心，半径为1.2000~cm；事件定位误差小于0.01~s。图~\\ref{fig:q4compare}显示收缩域含水率场以及三种模型终点。",
    "终点时未舍入的全域最大含水率刚好低于0.15~kg/kg，位于中心，半径为1.2000~cm；事件搜索区间宽度小于0.01~s。图~\\ref{fig:q4compare}比较附录4物性下收缩与固定半径模型，并给出收缩模型的中心、径向平均和表面含水率。",
)

section(
    "% TODO（图8源图）",
    "各传递参数的$S_p$均为负",
    "完整的参数扰动图移至附录~\\ref{app:numerical}，正文保留表~\\ref{tab:sensitivity}所需的定量结果。\n",
)

section(
    "\\subsubsection{终点判据和模型口径敏感性分析}",
    "\\subsubsection{数值收敛与守恒检验}",
    r"""\subsubsection{终点判据和模型口径敏感性分析}
采用$N=120$快速设置对不同口径作同精度比较，并把分段最大时间步连续减半到基准设置的$1/8$；各方案末两级终点差均小于0.007~h。表~\ref{tab:criterion-robust}中的变化量均相对同精度的快速基准计算，正式结果另列，因而不会把离散差异混入口径影响。
\begin{table}[htbp]
\centering\small
\caption{问题三、四不同实现口径的终点比较}\label{tab:criterion-robust}
\begin{tabular}{lrrrr}
\toprule
方案 & 问题三/h & 变化/h & 问题四/h & 变化/h\\\midrule
本文正式计算 & 57.47 & -- & 51.09 & --\\
$N=120$同精度基准 & 57.48 & 0 & 51.09 & 0\\
4~h后保持附件末值 & 57.18 & $-0.30$ & 50.83 & $-0.27$\\
径向平均含水率判据 & 36.03 & $-21.45$ & 34.97 & $-16.12$\\
半径采用分段线性插值 & -- & -- & 51.10 & $+0.004$\\\bottomrule
\end{tabular}
\end{table}
环境延拓会带来约十几分钟差异；平均含水率不能保证中心达标，会过早终止；在相同材料坐标模型内，PCHIP与线性半径插值的影响不足一分钟。由此可见，第三、四问的小时级偏差主要来自达标判据和移动边界建模口径，而不是事件搜索精度。

""",
)

section(
    "\\subsubsection{数值收敛与守恒检验}",
    "在物理合理性方面",
    r"""\subsubsection{数值收敛与守恒检验}
问题一、二空间加密的最大含水率差均小于$10^{-5}$~kg/kg；问题三、四的空间、时间及事件步长加密后，终点变化均小于0.1~s。将Picard容差收紧100倍，问题四含水率最大差为$4.09\times10^{-13}$~kg/kg。四问最大水量收支残差不超过$2.54\times10^{-12}$~kg/kg。详细加密表和收敛图移至附录~\ref{app:numerical}，避免数值验证挤占模型论证篇幅。

""",
)

required(
    "本参赛队在论文整理过程中使用了AI工具，主要用于LaTeX正文草拟、代码与结果文件的结构化核对、图表版式检查及编译错误排查。",
    "本参赛队在论文整理和修订过程中使用了AI工具，主要用于LaTeX正文草拟、评审意见整理与落实、代码和结果文件的结构化核对、图表版式检查及编译错误排查。",
)
required(
    "\\nolinkurl{analysis/sensitivity_analysis.py} & 单因素灵敏度计算、CSV导出与绘图 & 模型分析\\\\",
    "\\nolinkurl{analysis/sensitivity_analysis.py} & 单因素灵敏度计算、CSV导出与绘图 & 模型分析\\\\\n\\nolinkurl{recalculation/build_v02_assets.py}, \\nolinkurl{口径对比补算.csv} & 口径对照补算、步长加密与修订图件生成 & 稳健性分析\\\\",
)

insert = r"""
\section{详细数值验证}\label{app:numerical}
\begin{table}[htbp]
\centering\small
\caption{主要数值加密与守恒检验}
\begin{tabularx}{\textwidth}{l X r r}
\toprule
问题&检验方式&最大场差/(kg/kg)&终点差/s\\\midrule
1&$N=960\to1280$空间加密&$4.070\times10^{-6}$&--\\
2&$N=960\to1280$空间加密&$3.547\times10^{-6}$&--\\
3&$N=960\to1280$匹配前缀空间加密&--&0.0439\\
3&$\Delta t=0.0625\to0.03125$~s&$5.483\times10^{-7}$&0.0952\\
3&$N=1280\to1600$空间加密&$1.002\times10^{-6}$&0.0146\\
4&$\Delta t=0.0625\to0.03125$~s&$9.412\times10^{-7}$&0.0732\\
4&$N=1280\to1600$空间加密&$1.152\times10^{-5}$&0.0146\\\bottomrule
\end{tabularx}
\end{table}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.82\linewidth]{analysis/fig9_sensitivity.pdf}
  \caption{参数扰动对达标时间的影响；$h_m$表示传质系数}
\end{figure}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.82\linewidth]{q4/figures/fig8_combined_convergence.pdf}
  \caption{问题三、四空间和时间加密的终点收敛}
\end{figure}

"""
required("\n完整源程序代码及运行环境见支撑材料文件。\n", "\n完整源程序代码及运行环境见支撑材料文件。\n" + insert)

path.write_text(text, encoding="utf-8", newline="\n")
