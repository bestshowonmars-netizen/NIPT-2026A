"""Paper figure 8 from the saved Q3/Q4 refinement matrices; no numerical solves."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
from validation.evidence_paths import project_path
import hashlib
import json
from fractions import Fraction
import numpy as np
from common.plotting import BLUE, ORANGE, setup, save
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = bootstrap.ROOT


def _path(value, root):
    return project_path(value, root)


def _fraction(value):
    return str(Fraction(float(value)).limit_denominator(100000))


def _same(records, keys):
    for key in keys:
        first = records[0]["metadata"].get(key)
        if any(record["metadata"].get(key) != first for record in records[1:]):
            raise ValueError(f"Combined figure mixes the fixed parameter {key}")


def _series(root, question, kind):
    report_path = root / f"validation/q{question}_convergence_{kind}.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    runs = [run for run in report["runs"] if not run.get("tight", False)]
    if question == 3:
        required_source = "matched_q2" if kind == "space" else "production"
        runs = [run for run in runs if run.get("source") == required_source]
    records = []
    for run in runs:
        path = _path(run["path"], root)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
        if not meta.get("completed") or not run.get("checks", {}).get("passed"):
            raise ValueError(f"Figure 8 requires a complete, checked case: {path}")
        if float(meta["finish_time_s"]) != float(run["finish_time_s"]):
            raise ValueError(f"Stale endpoint in the refinement report: {path}")
        records.append({"path": path.resolve().as_posix(), "metadata": meta,
                        "cells": int(meta["cells"]), "dt": float(meta["dt_max"]),
                        "finish_time_s": float(meta["finish_time_s"]),
                        "requested_event_dt": float(run.get("requested_event_dt", run["event_dt"]))})
    if len(records) < 2:
        raise ValueError(f"At least two actual cases are required for Q{question} {kind}")
    _same(records, ("question", "model_question", "radius", "h", "hm", "initial_T", "initial_C",
                    "atol", "rtol", "max_iterations", "epsilon_t", "shrinking", "dt_scale", "time_growth"))
    if kind == "space":
        _same(records, ("dt_max", "event_dt"))
        if question == 3:
            seed_keys = ("dt_scale", "dt_max", "time_growth", "atol", "rtol", "radius", "h", "hm")
            first_seed = records[0]["metadata"]["source_metadata"]
            for record in records:
                seed = record["metadata"]["source_metadata"]
                if seed["cells"] != record["cells"] or any(seed.get(k) != first_seed.get(k) for k in seed_keys):
                    raise ValueError("Q3 spatial cases must use their own matching Q2 mesh and common startup settings")
        records.sort(key=lambda record: record["cells"])
        reference = records[-1]
    else:
        _same(records, ("cells",))
        if question == 3:
            _same(records, ("source_path",))
        event_caps = {record["requested_event_dt"] for record in records}
        if len(event_caps) != 1:
            raise ValueError("Time sweeps must share the requested event-step cap")
        for record in records:
            if record["metadata"]["event_dt"] != min(record["dt"], record["requested_event_dt"]):
                raise ValueError("Actual event step differs from its documented time-step clamp")
        records.sort(key=lambda record: record["dt"])
        reference = records[0]
    coordinates = [record["cells" if kind == "space" else "dt"] for record in records]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Figure 8 cannot collapse duplicate refinement coordinates")
    points = [{key: record[key] for key in ("path", "cells", "dt", "finish_time_s")}
              | {"deviation_s": abs(record["finish_time_s"] - reference["finish_time_s"])}
              for record in records]
    meta = records[0]["metadata"]
    settings = {"cells": meta["cells"] if kind == "time" else None,
                "dt_max": meta["dt_max"] if kind == "space" else None,
                "event_tol": meta["epsilon_t"],
                "event_dt": meta["event_dt"] if kind == "space" else None,
                "requested_event_dt": records[0]["requested_event_dt"],
                "dt_scale": meta.get("dt_scale"), "time_growth": meta.get("time_growth")}
    if question == 3:
        settings["q2_prefix"] = {key: meta["source_metadata"][key]
                                  for key in ("dt_scale", "dt_max", "time_growth")}
        settings["q2_prefix_mode"] = "own_matching_mesh" if kind == "space" else "same_production_state"
    return {"question": question, "kind": kind, "points": points, "settings": settings,
            "reference_path": reference["path"], "reference_finish_time_s": reference["finish_time_s"],
            "report_path": report_path.resolve().as_posix(),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}


def _latex_caption(series):
    q3s, q4s, q3t, q4t = (series[key]["settings"] for key in ("q3_space", "q4_space", "q3_time", "q4_time"))
    return (
        r"问题3与问题4的空间、时间离散对干燥终点的影响。纵轴为各模型相对本序列最细参考档的终点绝对偏差，采用线性轴；零点为参考档自比较，并非解析真解误差。"
        rf"空间序列分别固定$\Delta t_3={q3s['dt_max']:g}$ s、$\Delta t_4={q4s['dt_max']:g}$ s，均以本组最大$N$为参考；"
        rf"问题3各网格分别使用自身问题2末态，早期$\tau={_fraction(q3s['q2_prefix']['dt_scale'])}$ s；"
        rf"问题4从初态独立计算，$\tau={_fraction(q4s['dt_scale'])}$ s。"
        rf"时间序列分别固定$N_3={q3t['cells']}$、$N_4={q4t['cells']}$，均以本组最小时间步为参考；"
        r"问题3共用同一正式问题2末态，问题4保持相同初期步长规则。"
        rf"时间序列事件步长均取$\min(\Delta t_{{\max}},{q3t['requested_event_dt']:g}\ \mathrm{{s}})$，"
        rf"事件夹逼容差分别为{q3t['event_tol']:g} s与{q4t['event_tol']:g} s。"
        r"空间含水率差仍需独立验收，不能仅凭本图的终点偏差接受网格。"
    )


def combined_convergence_figure(root=ROOT, folder=None, latex_path=None, receipt_path=None, plots=True):
    root = Path(root)
    series = {f"q{question}_{kind}": _series(root, question, kind)
              for kind in ("space", "time") for question in (3, 4)}
    selected_path = root / "validation/q4_selected_config.json"
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        time_series = series["q4_time"]
        if (time_series["settings"]["cells"] != selected["cells"]
                or min(point["dt"] for point in time_series["points"]) != selected["dt_max"]):
            raise ValueError("Q4 time curve does not contain the final selected refined configuration")
    if series["q3_time"]["settings"]["requested_event_dt"] != series["q4_time"]["settings"]["requested_event_dt"]:
        raise ValueError("Caption requires equal event-step caps across the two time sequences")
    folder = Path(folder) if folder else root / "q4/figures"
    if plots:
        setup()
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5), layout="constrained")
        for ax, kind in zip(axes, ("space", "time")):
            all_x = set()
            for question, color, marker in ((3, BLUE, "o"), (4, ORANGE, "s")):
                item = series[f"q{question}_{kind}"]
                key = "cells" if kind == "space" else "dt"
                xs = [point[key] for point in item["points"]]
                ys = [point["deviation_s"] for point in item["points"]]
                all_x.update(xs)
                fixed = (f"Δt={_fraction(item['settings']['dt_max'])} s" if kind == "space"
                         else f"N={item['settings']['cells']}")
                ax.plot(xs, ys, color=color, marker=marker, ls="-" if question == 3 else "--",
                        ms=4 if question == 3 else 6, mfc=color if question == 3 else "none",
                        mew=1.1, lw=1.2, label=f"问题{question}：{fixed}", clip_on=False)
            if kind == "time":
                ax.set_xscale("log", base=2)
                ticks = sorted(all_x)
                ax.set_xticks(ticks, [_fraction(t) for t in ticks])
            else:
                ax.set_xticks(sorted(all_x))
            ax.set_ylim(bottom=0)
            ax.set(title="(a) 空间离散" if kind == "space" else "(b) 时间离散",
                   xlabel="参考径向单元数 N" if kind == "space" else "时间步上限 Δt / s",
                   ylabel="相对本组最细参考的终点偏差 / s")
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(True)
            ax.legend(frameon=False, fontsize=7.1)
        fig.suptitle("相对于各自最细离散结果的终点偏差", fontsize=8)
        save(fig, folder, "fig8_combined_convergence")
    latex_path = Path(latex_path) if latex_path else root / "q4/数值验证配图.tex"
    caption = _latex_caption(series)
    text = "\n".join([
        r"% 正文图8在前，第四问场加密证据可放附录；主文档需加载graphicx、amsmath。",
        r"\begin{figure}[htbp]", r"  \centering",
        r"  \includegraphics[width=0.98\linewidth]{q4/figures/fig8_combined_convergence.pdf}",
        r"  \caption{" + caption + "}", r"  \label{fig:combined-convergence}", r"\end{figure}", "",
        r"各模型、各序列分别定义",
        r"\[E_t^{(q)}=\left|t_+^{(q)}-t_{\mathrm{ref}}^{(q)}\right|,\qquad q=3,4.\]",
        r"参考终点是对应离散序列的最细已计算结果，不是解析真解。局部事件括区、终点加密差与场量加密差分别核对，不将其合称为严格总误差上界。", "",
        r"% 以下补图可放第四问数值验证附录。",
        r"\begin{figure}[htbp]", r"  \centering",
        r"  \includegraphics[width=0.98\linewidth]{validation/figures/q4_convergence.pdf}",
        r"  \caption{第四问的时间步与场量加密检查。左图为实际时间步下的终点变化；右图为各网格相对更细参考网格的四类含水率最大差，虚线为$2\times10^{-5}$ kg/kg门槛。该补图保留未通过的960网格筛选记录，避免仅以终点差判断网格充分性。}",
        r"  \label{fig:q4-convergence}", r"\end{figure}", "",
    ])
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(text, encoding="utf-8", newline="\n")
    receipt_path = Path(receipt_path) if receipt_path else root / "validation/q4_combined_convergence.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"series": series, "caption": caption, "linear_deviation_axis": True,
               "reference_self_comparisons_are_zero": True, "exact_solution_error_claimed": False}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return {"figure": str(folder / "fig8_combined_convergence.png"), "latex": str(latex_path),
            "source_record": str(receipt_path)}
