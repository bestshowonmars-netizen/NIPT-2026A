"""Q4 paper text and validation appendix built from saved numerical evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
from validation.evidence_paths import project_path
import argparse
from collections import defaultdict
from fractions import Fraction
import csv
import json
import math
import numpy as np
from common.solver import load_solution
from common.plotting import BLUE, ORANGE, GRAY, setup, save
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = bootstrap.ROOT
MATRICES = ("time", "space", "startup", "picard", "event", "event_step")
KINDS = {"time": "时间步", "space": "空间网格", "space_cross_two_levels": "空间跨两档",
         "startup": "初期步长", "picard_tolerance": "Picard容差",
         "event_tolerance": "事件容差", "event_integration_step": "事件积分步长",
         "fixed_time": "固定A4时间步", "fixed_space": "固定A4空间网格"}


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def _path(value, root):
    return project_path(value, root)


def load_evidence(root=ROOT):
    root = Path(root)
    reports, runs, comparisons, evidence_paths = {}, {}, [], {}
    seen_comparisons = set()
    paths = [(name, root / "validation" / f"q4_convergence_{name}.json") for name in MATRICES]
    paths += [(path.stem.removeprefix("q4_convergence_"), path)
              for path in sorted((root / "validation").glob("q4_convergence_*.json"))
              if path.stem.removeprefix("q4_convergence_") not in MATRICES]
    paths.append(("space_screen", root / "validation/q4_space_screen.json"))
    paths += [(path.stem.removeprefix("q4_"), path)
              for path in sorted((root / "validation").glob("q4_fixed_*.json"))]
    for name, path in paths:
        data = _read_json(path)
        if data is None:
            continue
        items = list(data.get("comparisons", []))
        if isinstance(data.get("comparison"), dict) and data["comparison"] not in items:
            items.append(data["comparison"])
        if name.startswith("fixed_") and not items and not data.get("runs"):
            continue
        reports[name] = data
        evidence_paths[name] = path
        originals = list(data.get("runs", []))
        known_paths = {_path(run["path"], root).resolve() for run in originals}
        for item in items:
            for side in ("coarse", "fine"):
                case_path = item.get(side + "_path", item.get("case_paths", {}).get(side))
                if case_path and _path(case_path, root).resolve() not in known_paths:
                    originals.append({"label": item.get(side) or Path(case_path).stem, "path": case_path})
                    known_paths.add(_path(case_path, root).resolve())
        for original in originals:
            run = dict(original)
            meta = _read_json(_path(run["path"], root).with_suffix(".json")) or {}
            run["metadata"] = meta
            config = {**run.get("config", {}), **meta}
            for key, source in (("cells", "cells"), ("dt", "dt_max"), ("dt_scale", "dt_scale"),
                                ("event_dt", "event_dt"), ("event_tol", "epsilon_t"),
                                ("shrinking", "shrinking"), ("finish_time_s", "finish_time_s")):
                if source in config:
                    run[key] = config[source]
            run.setdefault("finish_time_s", run.get("checks", {}).get("finish_time_s"))
            runs[run["label"]] = run
        for item in items:
            comparison_identity = {key: item.get(key) for key in
                                   ("kind", "coarse_config", "fine_config", "physical_C", "reference_C",
                                    "surface_C", "full_grid_maximum_C", "C", "T", "finish_time_s", "targets")}
            for side in ("coarse", "fine"):
                saved_path = item.get(side + "_path", item.get("case_paths", {}).get(side))
                comparison_identity[side] = (_path(saved_path, root).resolve().as_posix()
                                             if saved_path else item.get(side))
            signature = json.dumps(comparison_identity, sort_keys=True, ensure_ascii=False)
            if signature not in seen_comparisons:
                comparisons.append({**item, "matrix": name})
                seen_comparisons.add(signature)
    ordered = sorted(runs.values(), key=lambda r: (not r.get("shrinking", True), int(r.get("cells", 0)),
                     -float(r.get("dt", 0)), -float(r.get("dt_scale", 0)),
                     -float(r.get("event_dt", 0)), -float(r.get("event_tol", 0)), r["label"]))
    labels = {r["label"]: f"Q{i + 1:02d}" for i, r in enumerate(ordered)}
    return {"root": root, "reports": reports, "runs": runs, "ordered_runs": ordered,
            "comparisons": comparisons, "labels": labels, "evidence_paths": evidence_paths,
            "labels_by_path": {_path(r["path"], root).resolve().as_posix(): labels[r["label"]] for r in ordered},
            "selected": _read_json(root / "validation/q4_selected_config.json"),
            "acceptance": _read_json(root / "validation/q4_acceptance.json"),
            "checks": _read_json(root / "validation/q4_checks.json"),
            "reproduction": _read_json(root / "validation/q4_reproduction.json")}


def _fraction(number):
    return str(Fraction(float(number)).limit_denominator(100000)) if number is not None else "未记录"


def _complete(solution):
    if solution is None:
        return False
    meta = solution["metadata"]
    return meta.get("completed") is True and meta.get("finish_time_s") is not None


def _table6_rows(root, table_path=None):
    path = Path(table_path) if table_path else root / "q4/tables/table6.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def paper_blocks(solution, control, q3, evidence):
    """Return shared prose/formula blocks for both Markdown and LaTeX output."""
    if not _complete(solution):
        raise ValueError("A completed Q4 state is required before writing numerical paper conclusions.")
    meta = solution["metadata"]
    data = np.asarray(solution["radius_data"])
    finish = float(meta["finish_time_s"])
    radius = float(solution["radius_m"][-1])
    bracket = meta["bracket"]
    maximum = float(solution["max_C"][-1])
    max_r = float(solution["max_r"][-1])
    blocks = [
        ("heading", "半径数据与材料坐标"),
        ("paragraph", f"附件2提供{len(data)}个半径观测点，时间从{data[0, 0] / 3600:g} h至{data[-1, 0] / 3600:g} h，"
         f"对应初始半径{data[0, 1] * 100:g} cm与末端观测半径{data[-1, 1] * 100:.3f} cm。采用全部节点的保形分段三次Hermite插值（PCHIP）建立$R(t)$，逐段保留观测平台；"
         "插值表示实际送入模型的半径函数，不额外拟合收缩机理。若计算超过观测范围，延用最后观测半径，并将这一处理记作边界延拓约定。"),
        ("equation", r"R_0=0.02\ \mathrm{m},\qquad s(t)=\frac{R(t)}{R_0},\qquad x=\frac{r}{s(t)}\in[0,R_0],\qquad \xi=\frac{x}{R_0}=\frac{r}{R(t)}."),
        ("paragraph", r"按给定的径向同比收缩假设，同一材料位置满足$r=s(t)x$，运动学速度为$v_r(r,t)=\dot s(t)r/s(t)$。因此固定$x$或$\xi$随材料运动。长度保持不变，一维模型忽略轴向与端部差异；这一运动学假设使物理收缩区域对应到固定参考区域。实际材料坐标求解只需评价$R(t)$，无需对半径观测求数值导数或另加显式对流项。"),
        ("figure", ("fig6_q4_radius_mapping", "药材收缩半径输入与材料坐标映射", "fig:q4-radius")),
        ("heading", "附录4物性与参考域方程"),
        ("paragraph", r"第四问从题设均匀初态$t=0$独立计算，全部阶段使用附录4物性。温度$T$在物性公式中采用K；环境温度数据先由摄氏度转换为K。$\rho,c_p,k,D$的单位依次为$\mathrm{kg/m^3}$、$\mathrm{J/(kg\cdot K)}$、$\mathrm{W/(m\cdot K)}$与$\mathrm{m^2/s}$，$C$为干基含水率，单位kg/kg。"),
        ("equation", r"\begin{aligned}\rho(C)&=760+90C,\\ c_p(C)&=1850+2150\frac{C}{C+1},\\ k(C)&=0.12+0.20\frac{C}{C+1},\\ D(C,T)&=4.2\times10^{-4}\exp\!\left(-\frac{0.30}{C}\right)\exp\!\left(-\frac{3850}{T}\right).\end{aligned}"),
        ("paragraph", "材料坐标消去由同比运动产生的显式对流项，径向扩散算子出现$s^{-2}$缩放。本文沿用有效传热传质模型，在$0<x<R_0$上求解"),
        ("equation", r"\begin{aligned}\rho(C)c_p(C)\left.\frac{\partial T}{\partial t}\right|_x&=\frac1x\frac{\partial}{\partial x}\left[x\frac{k(C)}{s(t)^2}\frac{\partial T}{\partial x}\right],\\ \left.\frac{\partial C}{\partial t}\right|_x&=\frac1x\frac{\partial}{\partial x}\left[x\frac{D(C,T)}{s(t)^2}\frac{\partial C}{\partial x}\right].\end{aligned}"),
        ("paragraph", "中心采用对称条件，表面采用与物理域等价的Robin交换条件。将方程通量系数写成$k/s^2$、$D/s^2$时，参考域的对流换热、传质系数必须同时写成$h/s$、$h_m/s$："),
        ("equation", r"\begin{aligned}T_x(0,t)&=C_x(0,t)=0,\\ -\frac{k}{s^2}T_x(R_0,t)&=\frac{h}{s}\bigl(T_s-T_\infty\bigr),\\ -\frac{D}{s^2}C_x(R_0,t)&=\frac{h_m}{s}\bigl(C_s-C_\infty^{\mathrm{eff}}\bigr),\\ T(x,0)&=301.15\ \mathrm{K},\qquad C(x,0)=2.55\ \mathrm{kg/kg}.\end{aligned}"),
        ("paragraph", r"上述$s$因子来自坐标变换，实际物性仍由附录4给定。表面对流系数按既定假设沿用附录2，即$h=25\ \mathrm{W/(m^2\cdot K)}$、$h_m=8\times10^{-7}\ \mathrm{m/s}$。控制体的体积与距离始终采用初始参考几何，缩放仅施加于内部输运和表面交换系数；有效热存储系数仍为$\rho(C)c_p(C)$。"),
        ("paragraph", "环境输入在0—4 h内采用附件1的分段线性插值，之后采用既定的50.0℃与0.0500 kg/kg恒值延拓。环境水分指标仍按药材侧等效驱动解释。模型不显式计入潜热；经验密度用于热容量。初始干物质分布均匀、各材料壳干物质质量保持不变，因此初始参考体积权重同时给出固定干质量权重，水量收支始终使用这一归一化口径。"),
        ("heading", "有限体积离散与水量核对"),
        ("paragraph", r"参考域使用向表面加密的原生网格，控制体通量在相邻单元间共享；时间推进采用后向Euler全隐式格式与Picard耦合迭代，并同时检查迭代变化与离散残差。每一步在新的绝对时刻评价环境、半径和物性，并对齐输入节点、输出时刻及事件候选时刻。省略所有控制体共有的圆周与长度因子$2\pi\ell$后，径向体积因子及归一化含水率写为"),
        ("equation", r"V_j=\frac{x_{j+1/2}^2-x_{j-1/2}^2}{2},\qquad V_0=\sum_jV_j=\frac{R_0^2}{2},\qquad \overline C^n=\frac{\sum_jV_j C_j^n}{V_0}."),
        ("paragraph", r"记$x_j$为参考控制体中心，$s_{n+1}=s(t_{n+1})$。在新的时刻定义$\widehat D_j=D(C_j^{n+1},T_j^{n+1})/s_{n+1}^2$、$\widehat h_m=h_m/s_{n+1}$，全隐式含水率离散式为"),
        ("equation", r"V_j\frac{C_j^{n+1}-C_j^n}{\Delta t_n}=G_{j-1/2}\bigl(C_{j-1}^{n+1}-C_j^{n+1}\bigr)+G_{j+1/2}\bigl(C_{j+1}^{n+1}-C_j^{n+1}\bigr),"),
        ("equation", r"G_{j+1/2}=\frac{x_{j+1/2}}{\displaystyle\frac{x_{j+1/2}-x_j}{\widehat D_j}+\frac{x_{j+1}-x_{j+1/2}}{\widehat D_{j+1}}},\quad 1\le j<N;\qquad G_s=\frac{R_0}{\displaystyle\frac{R_0-x_N}{\widehat D_N}+\frac1{\widehat h_m}}."),
        ("paragraph", r"中心边界取$G_{1/2}=0$；末单元右侧取$G_{N+1/2}=G_s$，并将外侧值替换为$C_\infty^{\mathrm{eff}}(t_{n+1})$。热量方程具有相同通量结构，将$C$替换为$T$，左侧$V_j$替换为$a_jV_j$，其中$a_j=\rho(C_j^{n+1})c_p(C_j^{n+1})$；内部与表面系数分别用$\widehat k_j=k(C_j^{n+1})/s_{n+1}^2$和$\widehat h=h/s_{n+1}$构造，外侧值为$T_\infty(t_{n+1})$。每轮Picard迭代按当前温度、含水率更新物性与上述界面系数，直到状态变化和离散残差同时满足容差。"),
        ("paragraph", "干基含水率跟随材料位置，不因体积收缩额外乘以$s^{-2}$。采用正向为流出材料的边界水分通量，参考归一化累计失水与收支残差定义为"),
        ("equation", r"L(t)=\frac1{V_0}\int_0^t\frac{R_0h_m}{s(\tau)}\bigl[C_s(\tau)-C_\infty^{\mathrm{eff}}(\tau)\bigr]\,\mathrm d\tau,\qquad B(t)=\overline C(t)+L(t)-2.55."),
        ("paragraph", f"数值累计失水使用实际接受时间步的离散表面通量，与有限体积状态同步累加。本次末态平均含水率为{float(solution['mean_C'][-1]):.4f} kg/kg，"
         f"累计失水为{float(solution['loss_cumulative'][-1]):.4f} kg/kg；全程最大水量收支残差为{float(np.max(np.abs(solution['water_balance']))):.6e} kg/kg。该残差用于核对所采用离散模型的收支一致性。"),
        ("heading", "全域达标判据与实际位置取样"),
        ("equation", r"M_4(t)=\max_{0\le r\le R(t)}C_4(r,t),\qquad g_4(t)=M_4(t)-0.15,\qquad g_4(t_-)\ge0,\quad g_4(t_+)<0."),
        ("paragraph", "每个候选时刻检查全部原生单元，并加入中心对称重构值与真实移动表面的Robin重构值，同时记录实际最大值位置。定位采用同一未达标完整锚点的重新积分；严格条件以未舍入的$M_4<0.15$判断，不以固定21个表格点或101个绘图点的最大值替代。"),
        ("paragraph", f"本次严格达标时刻为$t_+={finish!r}$ s，即{finish / 3600:.4f} h；终点半径为{radius * 100:.4f} cm。"
         f"实际最大含水率为{maximum:.17g} kg/kg，位置为$r={max_r * 100:.4f}$ cm。"
         f"未达标—达标括区为[{float(bracket['t_minus'])!r}, {float(bracket['t_plus'])!r}] s，宽度{float(bracket['width']):.10g} s。"
         "括区宽度表示事件定位分辨率；离散加密差值与其分别核对，不据此宣称未知精确解的总误差上界。"),
        ("paragraph", r"表6按每6 h及精确终点列出固定实际位置$r=0,0.5,1,1.5,2$ cm的含水率，并独立列出表面值。result4.xlsx从60 s开始，以60 s为间隔输出$r=0,0.1,\ldots,2$ cm及真实表面。"
         "依据未舍入的半径判断实际位置：对于$r_j>R(t_n)$的位置，Excel、正文表格保留空白，JSON使用null；这些位置属于材料域外。表面列始终取$r=R(t_n)$，不由最外侧固定位置列代替。精确的非整分终点单独保存在终点文件与表6末行。所有取数来自同一份未舍入数值解，表格中的含水率统一作四位小数展示。"),
        ("table6", None),
        ("heading", "同物性固定半径对照"),
    ]
    if _complete(control):
        cm = control["metadata"]
        if cm.get("shrinking") is not False or cm.get("material_model") != "appendix4":
            raise ValueError("The shrinkage control must use Appendix 4 and a fixed radius.")
        for key in ("initial_T", "initial_C", "h", "hm"):
            if cm.get(key) != meta.get(key):
                raise ValueError(f"Unmatched physical control parameter: {key}")
        fixed_finish = float(cm["finish_time_s"])
        delta = finish - fixed_finish
        percentage = 100 * delta / fixed_finish
        direction = "提前" if delta < 0 else "延后" if delta > 0 else "相同"
        blocks += [("equation", r"\Delta t_{\mathrm{shrink}\mid\mathrm{A4}}=t_+(\mathrm{A4},R(t))-t_+(\mathrm{A4},R_0)."),
                   ("paragraph", f"附录4固定半径对照同样从$t=0$独立求解，采用相同初值与环境边界口径，其严格达标时刻为{fixed_finish / 3600:.3f} h。"
                    f"两组终点的有符号差值为{delta / 3600:.3f} h，相对固定对照为{percentage:.3f}%。"
                    f"在本组附录4模型条件下，收缩方案的模型终点{direction}。这一比较限定于同一物性下的几何与同比运动设定；问题3到问题4还同时改变了物性，不能将二者全部时长差异归因于收缩。"),
                   ("paragraph", rf"用于此处对照的实际数值配置为：收缩组$N={meta['cells']}$、$\Delta t_{{\max}}={meta['dt_max']:g}$ s；"
                    rf"固定组$N={cm['cells']}$、$\Delta t_{{\max}}={cm['dt_max']:g}$ s。各自的加密记录与比较范围在数值验证附录中列明。"),
                   ("paragraph", r"两组含水率加密差的核验门槛均为$2\times10^{-5}$ kg/kg；主收缩模型的终点差门槛为0.1 s，固定半径对照的终点差门槛单独取1 s。因此固定对照时长以0.001 h展示。门槛针对所列数值配对的实际差值，不代表未知真解的误差上界。")]
    else:
        blocks.append(("paragraph", "附录4固定半径对照尚未形成完整结果，因此这里只陈述第四问本身的结果，不量化收缩导致的时长变化。"))
    if _complete(q3):
        blocks.append(("paragraph", f"问题3附录3固定半径模型的终点为{float(q3['metadata']['finish_time_h']):.4f} h。图7将三组模型放在同一物理时间轴下比较；问题3前3 h未保存原生全域最大值，因此对应曲线保留空缺。"))
    blocks += [("figure", ("fig7_q4_shrinking_comparison", "移动区域内含水率演化与模型终点对照", "fig:q4-field")),
               ("paragraph", r"图7的场图由完整参考位置及其实际$r=\xi R(t)$坐标绘制，色块终止于真实表面，材料域外不参与色标、最大值或水量计算。阈值等值线与终点标记使用保存的真实数值；曲线与事件试算不作新根拟合。上述结果是所采用模型的预测与数值复核，附件没有提供内部实测温湿场，因而不将其表述为实验验证。")]
    acceptance = evidence.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
        blocks.append(("paragraph", "当前数值对应已保存的完整运行；正式选档的独立验收记录尚未确认，本文数值结论仍需与最终验收文件一致后定稿。"))
    return blocks


def _render_table(rows, latex=False):
    if not rows:
        return "表6尚未导出。"
    positions = ["r=0 cm", "r=0.5 cm", "r=1 cm", "r=1.5 cm", "r=2 cm"]
    if latex:
        lines = [r"\begin{table}[htbp]\centering", r"\caption{药材干燥过程的干基含水率（kg/kg）}\label{tab:q4-c}",
                 r"\begin{tabular}{c|rrrrrrr}\hline", r"时间/h & 0 cm & 0.5 cm & 1 cm & 1.5 cm & 2 cm & 药材表面 & $R(t)$/cm \\\hline"]
        for row in rows:
            label = f"{float(row['time_h']):.4f}（终点）" if row["row_kind"] == "endpoint" else f"{float(row['time_h']):g}"
            lines.append(" & ".join([label] + [row[p] for p in positions] + [row["surface_C"], f"{float(row['radius_cm']):.4f}"]) + r" \\")
        return "\n".join(lines + [r"\hline\end{tabular}", r"\end{table}"])
    lines = ["| 时间 / h | 0 cm | 0.5 cm | 1 cm | 1.5 cm | 2 cm | 药材表面 | R(t) / cm |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = f"{float(row['time_h']):.4f}（终点）" if row["row_kind"] == "endpoint" else f"{float(row['time_h']):g}"
        lines.append("| " + " | ".join([label] + [row[p] for p in positions] + [row["surface_C"], f"{float(row['radius_cm']):.4f}"]) + " |")
    return "\n".join(lines)


def render_paper(blocks, rows, root, latex=False):
    lines = ([r"% 论文正文片段：主文档需加载ctex、amsmath、graphicx；图路径相对于项目根目录。",
              r"\section{考虑径向收缩的药材烘干模型}"] if latex else ["# 问题4：考虑径向收缩的药材烘干模型"])
    for kind, value in blocks:
        if kind == "heading":
            text = "\\subsection{" + value + "}" if latex else "## " + value
        elif kind == "equation":
            text = "\\[\n" + value + "\n\\]"
        elif kind == "figure":
            stem, caption, label = value
            if latex:
                text = "\n".join([r"\begin{figure}[htbp]\centering", "\\includegraphics[width=0.98\\linewidth]{q4/figures/" + stem + ".pdf}",
                                   "\\caption{" + caption + "}\\label{" + label + "}", r"\end{figure}"])
            else:
                text = f"![{caption}]({(root / 'q4/figures' / (stem + '.png')).resolve().as_posix()})"
        elif kind == "table6":
            text = _render_table(rows, latex)
        else:
            text = value.replace("%", r"\%") if latex else value
        lines.extend(["", text])
    return "\n".join(lines) + "\n"


def _comparison_row(item, evidence):
    def label(side):
        saved = item.get(side)
        if saved in evidence["labels"]:
            return evidence["labels"][saved]
        path = item.get(side + "_path", item.get("case_paths", {}).get(side))
        if path:
            match = evidence["labels_by_path"].get(_path(path, evidence["root"]).resolve().as_posix())
            return match or saved or Path(path).stem
        return saved or "缺少算例来源"
    left, right = label("coarse"), label("fine")
    kind = KINDS.get(item.get("kind"), item.get("kind", "未记录"))
    if item.get("unavailable_run") or "C" not in item:
        return f"| {kind} | {left}→{right} | — | — | — | — | — | — | 未完成 |"
    fields = [f"{float(item[name]['maximum']):.5e}" for name in
              ("physical_C", "reference_C", "surface_C", "full_grid_maximum_C")]
    limit = item["targets"]
    return "| " + " | ".join([kind, f"{left}→{right}", *fields,
        f"{float(item['finish_time_s']['difference']):.9g}",
        f"{float(limit['full_output_C']):g} / {float(limit['endpoint_s']):g}",
        "通过" if item.get("passed") is True else "未通过"]) + " |"


def render_validation(evidence, figures=None):
    lines = ["# 问题4数值验证附录", "", "本文件只汇总已保存运行与独立比较记录，不启动数值模拟。",
             "", "## 比较量与域外规则", "",
             r"以共同整60秒时刻分别核对21个固定物理位置、101个参考位置、真实表面和原网格全域最大值，定义",
             "", r"\[E_C=\max\{E_{C,\mathrm{physical}},E_{C,\mathrm{reference}},E_{C,\mathrm{surface}},E_M\},\qquad E_t=|t_{+,a}-t_{+,b}|.\]", "",
             "固定物理位置只在两组相同的有效域内比较，先核对半径历史和逐点mask完全一致；域外NaN不能静默消除某一组内域误差。参考位置与真实表面单独核对，不由最外侧固定输出点代替。",
             "", r"主收缩模型的比较门槛为$E_C\le2\times10^{-5}$ kg/kg、$E_t\le0.1$ s；附录4固定半径控制计算用于评估整体时长效应，单独采用1 s终点差门槛。事件局部比较及其他检查使用各自记录的实际限值。加密差值与事件括区宽度不构成未知精确解的联合误差上界。", ""]
    acceptance = evidence.get("acceptance")
    required = (acceptance or {}).get("comparisons", (acceptance or {}).get("comparisons_recomputed_from_raw_arrays", []))
    lines += ["## 正式选档核验", ""]
    if required:
        passed = acceptance.get("passed") is True and all(item.get("passed") is True for item in required)
        lines += [f"原数组重算的必需比较{'全部通过' if passed else '尚未全部通过'}。此处按选中配置的必需对照判断，历史粗档失败保留在后表。", ""]
    else:
        lines += ["尚无可读取的正式选档原数组核验列表，因此不以选档文件的自述标志代替验收。", ""]
    header = ["| 检查 | 对照 | 物理C差 | 参考C差 | 表面C差 | 全域M差 | 终点差/s | 限值C/终点s | 结论 |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
    if required:
        lines += header + [_comparison_row(item, evidence) for item in required] + [""]
    lines += ["## 已保存算例", "", "| 编号 | N | Δt上限/s | 初期τ/s | 事件步/s | 夹逼容差/s | 半径模式 | 终点/s |", "|---|---:|---:|---|---:|---:|---|---:|"]
    for run in evidence["ordered_runs"]:
        lines.append(f"| {evidence['labels'][run['label']]} | {run.get('cells')} | {run.get('dt')} | {_fraction(run.get('dt_scale'))} | "
                     f"{run.get('event_dt')} | {run.get('event_tol')} | {'收缩' if run.get('shrinking') else '固定'} | {float(run['finish_time_s']):.9f} |")
    lines += ["", "每个常规加密算例均从题设初态在自身原网格上独立求解；无问题3末态拼接或网格间初态投影。局部事件对照则从相同完整锚点重积分，不能用共享前缀的零差值冒充全程收敛。", "",
              "## 全部实际加密记录", "", *header]
    lines += [_comparison_row(item, evidence) for item in evidence["comparisons"]]
    lines += ["", "表中含水率差的单位均为kg/kg；各自终点的参考剖面差发生在各自的终点时刻，单独保存，不能直接代替同一物理时刻的场误差。", ""]
    for item in evidence["comparisons"]:
        if (item.get("C_passed") is False
                and item.get("kind", "").removeprefix("fixed_").startswith("space")):
            view = max(("physical_C", "reference_C", "surface_C", "full_grid_maximum_C"),
                       key=lambda key: item[key]["maximum"])
            detail = item[view]
            location = (f"、ξ={detail['xi']:g}" if "xi" in detail else
                        f"、r={detail['radius_cm']:g} cm" if "radius_cm" in detail else "")
            model = "主收缩模型" if item["coarse_config"]["shrinking"] else "附录4固定半径对照"
            lines += [f"{model}保留了N={item['coarse_config']['cells']}→{item['fine_config']['cells']}的网格筛选失败："
                      f"最大含水率加密差{float(detail['maximum']):.9e} kg/kg出现在t={float(detail['time_s']):g} s{location}，"
                      f"超过{item['targets']['full_output_C']:g} kg/kg门槛。不能仅凭终点时间差接受该网格，正式选档以此前列出的必需比较为依据。", ""]
    missing = [name for name in MATRICES if name not in evidence["reports"]]
    if missing:
        lines += ["尚缺比较文件：" + "、".join(missing) + "。", ""]
    failures = [(name, label, detail) for name, report in evidence["reports"].items()
                for label, detail in report.get("failed_runs", {}).items()]
    if failures:
        lines += ["## 未完成的计算记录", ""]
        lines += [f"- {name} / {label}：{detail.get('type', '')}，{detail.get('message', '')}。"
                  for name, label, detail in failures]
        lines.append("")
    if figures is not None:
        combined = evidence["root"] / "q4/figures/fig8_combined_convergence.png"
        if combined.is_file():
            lines += ["## 问题3与问题4的终点加密总图", "",
                      f"![问题3与问题4的空间及时间终点加密]({combined.resolve().as_posix()})", "",
                      "各问题在各自参数匹配的空间、时间序列内选择最细参考；参考档的自比较零值保留在线性纵轴上。"
                      "问题3空间序列使用各网格自身的问题2末态，时间序列共用正式问题2末态；问题4各常规算例均从初态独立计算。"
                      f"全部点值、固定配置与来源记录见[q4_combined_convergence.json]({(evidence['root'] / 'validation/q4_combined_convergence.json').resolve().as_posix()})。", ""]
        lines += ["## 时间步与网格加密结果", "", f"![第四问终点变化与场加密差]({(Path(figures) / 'q4_convergence.png').resolve().as_posix()})",
                  "", "左图仅绘实际保存的时间步配置与终点，各点之间未补造缺失档位；事件积分步长按实际值分组。"
                  "右图以更细网格的数值结果作共同参考，展示固定物理点、参考位置、真实表面及全域最大值四类含水率差的最大值。"
                  "图中的通过标记仅指含水率差门槛，空间终点差仍见数值表。", ""]
    checks = evidence.get("checks")
    lines += ["## 正式文件与收支核对", ""]
    if isinstance(checks, dict):
        lines += ["以下数值直接读取正式检查记录。位置数按每一场量的全部保存时刻计数，Excel的规则分钟输出另表统计。", "",
                  "| 模型 | 检查结论 | 最大水量收支残差 / (kg/kg) | 域内物理位置数 | 域外物理位置数 | 参考位置数 |",
                  "|---|---|---:|---:|---:|---:|"]
        for key, name in (("solution", "附录4收缩"), ("fixed_control", "附录4固定")):
            detail = checks.get(key)
            if not isinstance(detail, dict):
                continue
            domains = detail["field_domains"]
            lines.append(f"| {name} | {'通过' if detail.get('passed') is True else '未通过'} | "
                         f"{float(detail['maximum_water_balance_residual']):.9e} | "
                         f"{domains['physical_in_domain_values']} | {domains['physical_outside_values']} | "
                         f"{domains['reference_values_per_field']} |")
        lines.append("")
        artifacts = checks.get("deliverables")
        if isinstance(artifacts, dict):
            lines += ["| 正式输出核对量 | 实际数量或结论 |", "|---|---:|"]
            for key, label in (("excel_regular_rows_checked", "Excel整60秒数据行"),
                               ("excel_physical_C_values_checked", "固定位置域内含水率值"),
                               ("excel_exterior_blank_cells_checked", "固定位置域外空白"),
                               ("excel_separate_surface_values_checked", "独立真实表面含水率值"),
                               ("table6_rows_checked", "表6数据行")):
                lines.append(f"| {label} | {artifacts[key]} |")
            lines += [f"| 精确终点文件核对 | {'通过' if artifacts.get('exact_endpoint_checked') is True else '未核对'} |",
                      f"| 导出整体结论 | {'通过' if artifacts.get('passed') is True else '未通过'} |", ""]
        core = checks.get("core")
        if isinstance(core, dict):
            lines += [f"基础离散核对{'通过' if core.get('passed') is True else '未通过'}："
                      f"核对了{core['radius_knots_checked']}个半径输入节点及{core['exact_plateau_intervals_checked']}个精确平台区间；"
                      f"插值越过相邻观测范围的最大值为{float(core['maximum_interpolation_overshoot_m']):.3e} m。", "",
                      "| 固定缩放下的独立稠密参考 | 温度最大差 / K | 含水率最大差 / (kg/kg) | 累计失水差 / (kg/kg) |",
                      "|---|---:|---:|---:|"]
            for scale, detail in core["fixed_radius_dense_reference"].items():
                lines.append(f"| {scale} | {float(detail['T_max_difference']):.6e} | "
                             f"{float(detail['C_max_difference']):.6e} | {float(detail['loss_difference']):.6e} |")
            restart = core["restart"]
            lines += ["", f"相同时间步边界下保存—重载的最大差分别为：温度{float(restart['T']):.6e} K，"
                      f"含水率{float(restart['C']):.6e} kg/kg，累计失水{float(restart['loss']):.6e} kg/kg。"
                      "封闭均匀场、移动表面重构与内部最大值位置的检查结论，以及完整来源散列均保留在原JSON中。", ""]
        lines += [f"完整检查记录：[q4_checks.json]({(evidence['root'] / 'validation/q4_checks.json').resolve().as_posix()})。", ""]
    else:
        lines += ["正式q4_checks.json尚不存在，导出逐格核对尚未确认。", ""]
    reproduction = evidence.get("reproduction")
    if isinstance(reproduction, dict):
        lines += ["## 附录独立运行复现", ""]
        if reproduction.get("passed") is True:
            lines += ["完整附录运行与正式结果的复现核对通过。附录从题设均匀初态独立推进第四问全程，未读取问题3末态作为初值；"
                      "核对对象是所采用同一算法的独立运行一致性，并非新的物理模型验证或内部实测验证。", "",
                      f"两次运行的终点均为{float(reproduction['finish_time_s']):.14g} s。"
                      "检查程序对终点时间、事件括区、全部保存时刻以及21个固定物理位置的域内外掩码要求完全相同；"
                      "原生网格的完整终态与其他未舍入场量逐项比较，限值为1×10⁻¹²。", "",
                      r"\[E_{\mathrm{reproduce}}(u)=\max_i|u_i^{\mathrm{appendix}}-u_i^{\mathrm{formal}}|.\]", "",
                      "| 未舍入核对量 | 最大绝对差 | 单位 |", "|---|---:|---|"]
            descriptions = {
                "final_T": ("完整原生网格终态温度", "K"),
                "final_C": ("完整原生网格终态含水率", "kg/kg"),
                "radius_m": ("全部保存时刻的半径", "m"),
                "surface_temperature_K": ("真实表面温度", "K"),
                "surface_moisture": ("真实表面含水率", "kg/kg"),
                "max_C": ("原生网格全域最大含水率历史", "kg/kg"),
                "max_r": ("实际最大含水率位置历史", "m"),
                "water_balance": ("水量收支残差历史", "kg/kg"),
                "loss_cumulative": ("累计失水历史", "kg/kg"),
                "temperature_K": ("固定21个物理位置的域内温度", "K"),
                "moisture": ("固定21个物理位置的域内含水率", "kg/kg"),
            }
            for key, value in reproduction.get("maximum_differences", {}).items():
                label, unit = descriptions.get(key, (key, ""))
                lines.append(f"| {label} | {float(value):.9e} | {unit} |")
            count = int(reproduction["excel_rows_including_header"]) - 1
            lines += ["", f"两份result4.xlsx的{count}行数据逐格相同：共{count * 21}个固定位置单元（包括材料域外空白）"
                      f"及{count}个独立表面数值。域外空白参与比较，未将其改写为零或表面值。", "",
                      "来源链由附录求解文件、独立数值输出、正式数值输出及正式Excel的SHA-256记录固定：", "",
                      "| 来源文件 | SHA-256 |", "|---|---|"]
            for relative, digest in reproduction.get("source_hashes", {}).items():
                lines.append(f"| [{relative}]({_path(relative, evidence['root']).resolve().as_posix()}) | {digest} |")
            lines.append("")
        else:
            lines += ["附录复现记录未通过，不将其表述为独立复现成功。", ""]
    lines += ["## 证据索引", ""]
    for name in evidence["reports"]:
        path = evidence["evidence_paths"][name]
        lines.append(f"- [{path.name}]({path.resolve().as_posix()})")
    for name in ("q4_selected_config.json", "q4_acceptance.json", "q4_checks.json", "q4_reproduction.json"):
        path = evidence["root"] / "validation" / name
        if path.is_file():
            lines.append(f"- [{name}]({path.resolve().as_posix()})")
    for run in evidence["ordered_runs"]:
        lines.append(f"- {evidence['labels'][run['label']]}：`{run['label']}`；[{Path(run['path']).name}]({_path(run['path'], evidence['root']).resolve().as_posix()})。")
    return "\n".join(lines) + "\n"


def convergence_figure(evidence, folder):
    """Saved time-step endpoints and observed concentration refinement differences."""
    setup()
    available = [run for run in evidence["ordered_runs"] if run.get("finish_time_s") is not None]
    if not available:
        return False
    origin = math.floor(min(float(r["finish_time_s"]) for r in available) / 10) * 10
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.7), layout="constrained")
    groups = defaultdict(list)
    for original in evidence["reports"].get("time", {}).get("runs", []):
        run = evidence["runs"][original["label"]]
        groups[(run["cells"], run.get("dt_scale"), run.get("event_dt"),
                run.get("event_tol"), run.get("shrinking"))].append(run)
    for i, (config, runs) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
        runs.sort(key=lambda r: float(r["dt"]))
        label = f"N={config[0]}，τ={_fraction(config[1])} s\n事件步={config[2]:g} s"
        left.plot([r["dt"] for r in runs], [float(r["finish_time_s"]) - origin for r in runs],
                  "o-", color=[BLUE, ORANGE, "#547B68"][i % 3], ms=4, lw=1.1, label=label)
    if groups:
        ticks = sorted({float(r["dt"]) for group in groups.values() for r in group})
        left.set_xscale("log", base=2)
        left.set_xticks(ticks, [_fraction(t) for t in ticks])
        left.legend(frameon=False, fontsize=6.3)
    else:
        left.text(.5, .5, "尚无完成的比较记录", transform=left.transAxes, ha="center", color=GRAY)
    left.set(title="(a) 时间步与达标终点", xlabel="时间步上限 Δt / s",
             ylabel=f"(终点时间 − {origin:g} s) / s")
    left.yaxis.set_major_locator(MaxNLocator(5))
    left.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Every plotted error is an actual saved raw-array comparison.  The fine
    # grid is a numerical reference, never a zero-error exact-solution point.
    candidates = [item for item in evidence["comparisons"]
                  if item.get("kind", "").startswith("space")
                  and item.get("coarse_config", {}).get("shrinking") is True
                  and item.get("fine_config", {}).get("shrinking") is True
                  and item.get("physical_masks_identical") is True
                  and item.get("radius_histories_identical") is True
                  and "C" in item]
    if candidates:
        finest = max(int(item["fine_config"]["cells"]) for item in candidates)
        pairs = {}
        for item in candidates:
            if int(item["fine_config"]["cells"]) != finest:
                continue
            coarse, fine = item["coarse_config"], item["fine_config"]
            matched_keys = ("dt_max", "dt_scale", "time_growth", "event_dt", "epsilon_t", "atol", "rtol")
            if any(coarse.get(key) != fine.get(key) for key in matched_keys):
                continue
            key = (int(coarse["cells"]), tuple(coarse.get(k) for k in matched_keys))
            pairs[key] = item
        spatial_groups = defaultdict(list)
        for (cells, config), item in pairs.items():
            spatial_groups[config].append((cells, item))
        for i, (config, pairs_in_group) in enumerate(sorted(spatial_groups.items(), key=lambda x: str(x[0]))):
            pairs_in_group.sort(key=lambda value: value[0])
            xs = [cells for cells, _ in pairs_in_group]
            ys = [float(item["C"]["maximum"]) * 1e5 for _, item in pairs_in_group]
            right.plot(xs, ys, "s-", color=[ORANGE, BLUE, "#547B68"][i % 3],
                       ms=4.5, lw=1.1, label=f"Δt={config[0]:g} s，τ={_fraction(config[1])} s")
            for cells, item in pairs_in_group:
                value = float(item["C"]["maximum"]) * 1e5
                right.annotate(f"{value:.3f}\n{'通过' if item.get('C_passed') is True else '未通过'}",
                               (cells, value), xytext=(0, 7), textcoords="offset points",
                               ha="center", va="bottom", fontsize=7)
        right.axhline(2, color="#222222", ls="--", lw=1, label="含水率差门槛")
        right.set_xticks(sorted({key[0] for key in pairs}))
        if pairs:
            xs = sorted({key[0] for key in pairs})
            pad = max(60, (max(xs) - min(xs)) * .25)
            right.set_xlim(min(xs) - pad, max(xs) + pad)
            ymax = max(2, max(float(item["C"]["maximum"]) * 1e5 for item in pairs.values()))
            right.set_ylim(0, ymax * 1.30)
        right.legend(frameon=False, fontsize=6.3, loc="lower left")
        right.set(title=f"(b) 相对 N={finest} 的场加密差",
                  xlabel="参考径向单元数 N", ylabel=r"最大含水率差 / ($10^{-5}$ kg/kg)")
    else:
        right.text(.5, .5, "尚无完成的空间比较", transform=right.transAxes, ha="center", color=GRAY)
        right.set(title="(b) 场加密差", xlabel="参考径向单元数 N", ylabel="最大含水率差 / (kg/kg)")
    right.yaxis.set_major_locator(MaxNLocator(5))
    for ax in (left, right):
        ax.grid(True)
    fig.suptitle("各网格均从初态独立求解；加密差不代表未知真解误差", fontsize=8)
    save(fig, folder, "q4_convergence")
    return True


def build_report(root=ROOT, solution_path=None, control_path=None, q3_path=None,
                 output=None, paper_path=None, latex_path=None, plots=True, table_path=None):
    root = Path(root)
    evidence = load_evidence(root)
    solution_path = Path(solution_path) if solution_path else root / "q4/solution.npz"
    control_path = Path(control_path) if control_path else root / "q4/fixed_control.npz"
    q3_path = Path(q3_path) if q3_path else root / "q3/solution.npz"
    solution = load_solution(solution_path)
    control = load_solution(control_path) if control_path.is_file() else None
    q3 = load_solution(q3_path) if q3_path.is_file() else None
    output = Path(output) if output else root / "validation/问题4精度验证报告.md"
    paper_path = Path(paper_path) if paper_path else root / "q4/方法草稿与结果解释.md"
    latex_path = Path(latex_path) if latex_path else root / "q4/论文正文_问题4.tex"
    blocks = paper_blocks(solution, control, q3, evidence)
    candidates = ([Path(table_path)] if table_path else
                  [paper_path.parent / "tables/table6.csv", solution_path.parent / "tables/table6.csv",
                   root / "q4/tables/table6.csv"])
    rows = []
    for candidate in candidates:
        available_rows = _table6_rows(root, candidate)
        if (available_rows and available_rows[-1].get("row_kind") == "endpoint"
                and float(available_rows[-1]["time_s"]) == float(solution["metadata"]["finish_time_s"])):
            rows = available_rows
            break
    if not rows:
        raise ValueError("Export table6 from this same numerical endpoint before generating the paper.")
    for path in (output, paper_path, latex_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(render_paper(blocks, rows, root), encoding="utf-8", newline="\n")
    latex_path.write_text(render_paper(blocks, rows, root, True), encoding="utf-8", newline="\n")
    figures = root / "validation/figures"
    made = convergence_figure(evidence, figures) if plots else (figures / "q4_convergence.png").is_file()
    combined = None
    if (evidence.get("acceptance") or {}).get("passed") is True:
        from validation.q4_combined_plot import combined_convergence_figure
        combined = combined_convergence_figure(root, plots=plots)
    output.write_text(render_validation(evidence, figures if made else None), encoding="utf-8", newline="\n")
    return {"paper_markdown": str(paper_path), "paper_latex": str(latex_path), "validation": str(output),
            "matrices_read": list(evidence["reports"]), "full_fixed_control_present": _complete(control),
            "combined_figure": combined}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--q3", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--latex", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.solution, args.control, args.q3,
                                 args.output, args.paper, args.latex, not args.no_plots, args.table), ensure_ascii=False, indent=2))
