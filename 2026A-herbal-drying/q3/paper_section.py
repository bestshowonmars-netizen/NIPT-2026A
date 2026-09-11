"""Generate a reusable LaTeX section from the verified Q3 arrays and records."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import json
from fractions import Fraction
import numpy as np
from common.solver import load_solution

ROOT = bootstrap.ROOT


def number(value):
    if value == 0:
        return "0"
    if abs(value) >= .001:
        return f"{value:.6g}"
    coefficient, exponent = f"{value:.3e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def write_section():
    solution = load_solution(ROOT/"q3/solution.npz")
    acceptance = json.loads((ROOT/"validation/q3_acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["passed"]
    metadata = solution["metadata"]
    t = metadata["finish_time_s"]
    bracket = metadata["bracket"]
    table_times = list(np.arange(21600., t, 21600.)) + [t]
    rows = []
    for second in table_times:
        i = int(np.flatnonzero(abs(solution["times"]-second) < 1e-8)[0])
        label = f"{second/3600:.4f}（终点）" if second == t else f"{second/3600:g}"
        cols = [int(np.argmin(abs(solution["radii"]-r))) for r in np.linspace(0, .02, 5)]
        rows.append(" & ".join([label]+[f"{x:.4f}" for x in solution["moisture"][i, cols]])+r" \\")
    descriptions = []
    for pair in acceptance["comparisons_recomputed_from_raw_arrays"]:
        if pair["kind"] == "time":
            phrase = rf"续算步长由 ${pair['coarse_config']['dt_max']:g}$ s 减至 ${pair['fine_config']['dt_max']:g}$ s"
        elif pair["kind"].startswith("space"):
            source = json.loads((ROOT/pair["coarse_path"]).with_suffix(".json").read_text(encoding="utf-8"))
            scale = Fraction(source["source_metadata"]["dt_scale"]).limit_denominator(100000)
            phrase = rf"空间对照固定续算步长 $\Delta t={pair['coarse_config']['dt_max']:g}$ s、各网格第二问前缀的时间尺度 $\tau=\frac{{{scale.numerator}}}{{{scale.denominator}}}$ s，原网格单元数由 ${pair['coarse_config']['cells']}$ 加至 ${pair['fine_config']['cells']}$"
        else:
            continue
        descriptions.append(phrase+rf"，共同输出位置的最大含水率差为 ${number(pair['C']['maximum'])}$ kg/kg，达标时间差为 ${pair['finish_time_s']['difference']:.8f}$ s。")
    boundary_path = ROOT/"validation/q3_boundary_sensitivity.json"
    boundary_note = ""
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        boundary_note = rf"在另一个匹配 $N=960,\Delta t=1$ s 的边界假设对照中，将4 h后的环境改为保持附件末值，得到总时长变化 ${boundary['signed_finish_change_s']/60:.4f}$ min（${boundary['relative_finish_change_percent']:.4f}\%$）。这体现输入延续假设的影响，不是离散误差或实验验证；正式结果仍采用事先确定的恒值延续。"
    content = rf"""% 由 q3/paper_section.py 从同一份已核验结果生成。
% 正文片段；在主文档导言区加载 amsmath、graphicx、booktabs，并支持中文。
% 从项目根目录编译时使用以下路径；若移动文件，相应调整图路径。
\section{{问题三：全域干燥达标时间}}

\subsection{{继承模型并读取完整末态}}
第二、三问的物性和尺寸保持一致，因此第三问是同一耦合模型的长时间续算。
在 $t_0=10800$ s 读取第二问全部 ${metadata['cells']}$ 个径向单元的未舍入温度、含水率及原网格，
维持固定半径 $R_0=0.02$ m，继续求解
\begin{{align}}
\rho(C)c_p(C)\frac{{\partial T}}{{\partial t}}&=
\frac1r\frac{{\partial}}{{\partial r}}\left(rk(C)\frac{{\partial T}}{{\partial r}}\right),\\
\frac{{\partial C}}{{\partial t}}&=
\frac1r\frac{{\partial}}{{\partial r}}\left(rD(C,T)\frac{{\partial C}}{{\partial r}}\right).
\end{{align}}
物性全部采用附录3，中心对称与表面对流条件沿用第二问。
内部继续计算温度场，通过当地温湿状态更新扩散系数，未将温度场直接固定为环境温度。
环境函数采用绝对时间：3--4 h仍按附件1插值，4 h之后按既定假设取
$T_\infty=323.15$ K、$C_\infty^{{\mathrm{{eff}}}}=0.0500$ kg/kg。

\subsection{{全域判据与终点定位}}
有限体积计算单元值与边界重构值共同构成全域最大含水率的数值判据：
\begin{{equation}}
M_h(t)=\max\left\{{C_{{\mathrm{{center}}}}^{{\mathrm{{rec}}}},C_1,\ldots,C_N,
C_{{\mathrm{{surface}}}}^{{\mathrm{{bc}}}}\right\}},\qquad
g_h(t)=M_h(t)-0.15.
\end{{equation}}
中心采用对称二次重构，表面值与Robin边界保持一致。
计算保留最大值的位置，严格达标条件为未舍入的 $g_h(t)<0$。
普通阶段采用全隐式时间推进和Picard迭代；检测到交叉后，从同一未达标完整状态
用细步分别重新积分至候选时刻，直至
\begin{{equation}}
g_h(t_-)\ge0,\quad g_h(t_+)<0,\quad t_+-t_-\le\varepsilon_t.
\end{{equation}}
报告达标侧 $t_+$，不以相邻输出值插值代替实际积分。
本次普通步长上限为 ${metadata['dt_max']:g}$ s，事件积分步长为 ${metadata['event_dt']:g}$ s，
夹逼容差为 ${metadata['epsilon_t']:g}$ s，得到
\begin{{align}}
[t_-,t_+]&=[{bracket['t_minus']:.11f},{bracket['t_plus']:.11f}]\ \mathrm{{s}},\\
t_{{\mathrm{{finish}}}}&={t:.11f}\ \mathrm{{s}}
\approx {t/3600:.4f}\ \mathrm{{h}}.
\end{{align}}
这是从题设初态起算的总时间；3 h后的续算时长为 ${t-10800:.8f}$ s。
末时刻全域最大值为 ${solution['max_C'][-1]:.17g}$ kg/kg，
对应位置为 $r={solution['max_r'][-1]*100:.4f}$ cm。

\subsection{{结果表与空间演化}}
表中数值统一保留四位小数。终点处显示的0.1500来自舍入，严格达标判定由上述未舍入结果给出。
完整结果文件按60 s与0.1 cm输出，非整分钟终点另存精确记录。
\begin{{table}}[htbp]
\centering
\caption{{每6小时及达标终点的干基含水率（kg/kg）}}
\begin{{tabular}}{{lrrrrr}}
\toprule
时间/h & $r=0$ cm & $r=0.5$ cm & $r=1$ cm & $r=1.5$ cm & $r=2$ cm\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\linewidth]{{q3/figures/fig5_q3_drying_endpoint.pdf}}
\caption{{全域最大含水率的阈值交叉及全过程含水率场。上幅从3 h的完整网格状态开始监测，
局部图为实际重新积分的事件试探点；下幅的0.15等值线展示各位置的达标顺序。}}
\end{{figure}}

\subsection{{数值验证与适用范围}}
以不变的初始干物质质量为基准，定义
\begin{{equation}}
w_i=\frac{{V_i}}{{\sum_jV_j}},\qquad
B(t)=\sum_iw_iC_i(t)+L_2+L_3(t)-2.55.
\end{{equation}}
上述表达用于 $t\ge10800$ s，其中 $L_2$ 来自第二问完整前3 h的累计失水，$L_3$ 来自续算段表面通量，均以初始干物质质量归一化。
前3 h的残差使用第二问各时刻自身的累计失水计算，不将完整 $L_2$ 提前计入。
本次全过程最大绝对水量残差为 ${number(float(np.max(abs(solution['water_balance']))))}$ kg/kg。
{chr(10).join(descriptions)}
空间比较使用各网格各自的第二问完整末态，未进行网格投影。
事件夹逼、步长加密和网格加密分别核查；这些差值不是未知精确解的严格误差上界。
第二问前3 h只保留了采样场历史，因而最大值曲线从3 h开始，未用早期采样最大值冒充全域最大值。

{boundary_note}
模型沿用等效环境水分驱动、无显式潜热项及径向主导等假设，所得结果为模型预测。
数值收支、独立附录复现及输入延续对照不代替内部温湿场实验验证。
"""
    path = ROOT/"q3/论文正文_问题3.tex"
    path.write_text(content, encoding="utf-8")
    (ROOT/"q3/数值验证配图.tex").write_text(r"""% 插入全文的数值验证部分，使图5之后的图6、图7位置保留给问题4。
\begin{figure}[htbp]
\centering
\includegraphics[width=0.96\linewidth]{q3/figures/fig8_q3_convergence.pdf}
\caption{第三问的时间步和空间网格比较。纵轴减去同一显示常数以便观察变化，该常数并非真解。
空间比较中的各网格均读取各自的第二问完整末态，正式第二问末态来源另作对照。}
\end{figure}
""", encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    write_section()
