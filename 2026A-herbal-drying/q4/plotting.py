"""Q4 input geometry and solved moving-domain figures; no synthetic solution data."""
from pathlib import Path
import json

import bootstrap
import numpy as np
from common.plotting import BLUE, ORANGE, GRAY, setup, save
from q4.shrinking_radius import radius_at
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

GREEN = "#547B68"
RED = "#B64545"


def _radius_curve(times, data, slopes):
    return np.asarray([radius_at(float(t), data, slopes, True, .02) for t in times])


def figure6(radius_data, radius_slopes, folder):
    """Plot all radius observations and the exact callable used by the solver."""
    setup()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    data = np.asarray(radius_data, dtype=float)
    slopes = np.asarray(radius_slopes, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or slopes.shape != (len(data),):
        raise ValueError("Figure 6 requires radius observations in seconds/metres and their solver PCHIP slopes.")
    fine = np.unique(np.r_[np.linspace(data[0, 0], data[-1, 0], 1801), data[:, 0]])
    hours, radius_cm = fine / 3600, _radius_curve(fine, data, slopes) * 100
    fig, (ax, physical, reference) = plt.subplots(1, 3, figsize=(8.6, 3.3), layout="constrained")

    ax.scatter(data[:, 0] / 3600, data[:, 1] * 100, color=GRAY, s=10,
               alpha=.6, zorder=2, label=f"观测（{len(data)}点）")
    ax.plot(hours, radius_cm, color=BLUE, lw=1.6, label="PCHIP插值", zorder=3)
    ax.set(xlabel="时间 / h", ylabel="药材半径 / cm")
    ax.set_title("(a) 半径输入", loc="left", fontsize=10)
    ax.set_xlim(hours[0], hours[-1])
    ax.set_ylim(float(data[:, 1].min() * 100) - .055, float(data[:, 1].max() * 100) + .055)
    ax.set_xticks([0, 24, 48, 72])
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.annotate(f"末值 {data[-1, 1] * 100:.3f} cm", xy=(hours[-1], radius_cm[-1]),
                xytext=(.48, .23), textcoords="axes fraction", fontsize=9, ha="center",
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": .8})

    physical.fill_between(hours, 0, radius_cm, color="#E7EEF4")
    physical.plot(hours, radius_cm, color=ORANGE, lw=1.6, label=r"表面 $r=R(t)$")
    for xi in (.25, .5, .75):
        physical.plot(hours, radius_cm * xi, color=BLUE, lw=1.0, ls=":")
    physical.set(xlabel="时间 / h", ylabel="实际半径 r / cm", ylim=(0, 2.1), xlim=(hours[0], hours[-1]))
    physical.set_title("(b) 物理坐标", loc="left", fontsize=10)
    physical.text(.55, .79, r"$r=\xi R(t)$", transform=physical.transAxes, fontsize=11, ha="center")
    reference.fill_between(hours, 0, 1, color="#E7EEF4")
    for xi in (.25, .5, .75):
        reference.axhline(xi, color=BLUE, lw=1.0, ls=":")
    reference.axhline(1, color=ORANGE, lw=1.6)
    reference.set(xlabel="时间 / h", ylabel=r"材料坐标 $\xi=r/R(t)$",
                  xlim=(hours[0], hours[-1]), ylim=(0, 1.05))
    reference.set_title("(c) 材料坐标", loc="left", fontsize=10)
    reference.text(.51, .61, r"固定 $\xi$ 随材料运动", transform=reference.transAxes,
                   ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    physical.set_yticks([0, .5, 1, 1.5, 2])
    reference.set_yticks([0, .25, .5, .75, 1])
    for current in (physical, reference):
        current.set_xticks([0, 24, 48, 72])
        current.grid(True, alpha=.7)
    save(fig, folder, "fig6_q4_radius_mapping")
    notes = {"figure": "fig6_q4_radius_mapping", "radius_observations": len(data),
             "input_last_time_s": float(data[-1, 0]), "input_last_radius_m": float(data[-1, 1]),
             "notes": ["全部观测点与求解器采用的同一PCHIP函数；没有另行拟合半径。",
                       "橙线为实际表面，蓝色虚线对应固定材料坐标xi=0.25、0.5、0.75。",
                       "三个完整面板共用0至附件末时刻的小时尺度，不再嵌入小字局部窗。"]}
    (Path(folder) / "fig6_plot_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_series(solution, comparisons):
    comparisons = comparisons or {}
    records = []
    if comparisons.get("q3") is not None:
        records.append(("问题3：附录3·固定半径", comparisons["q3"], BLUE, "--"))
    if comparisons.get("a4_fixed") is not None:
        records.append(("对照：附录4·固定半径", comparisons["a4_fixed"], "#65717A", "-."))
    records.append(("问题4：附录4·收缩半径", solution, ORANGE, "-"))
    return records


def figure7(solution, folder, comparison_solutions=None):
    """Show only saved fields inside the moving domain and actual stopping evidence."""
    setup()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    times = np.asarray(solution["times"], dtype=float)
    xi = np.asarray(solution["xi"], dtype=float)
    radius = np.asarray(solution["radius_m"], dtype=float)
    values = np.asarray(solution["moisture_ref"], dtype=float)
    physical_radii = np.asarray(solution["reference_physical_radii"], dtype=float)
    if values.shape != (len(times), len(xi)) or physical_radii.shape != values.shape:
        raise ValueError("Figure 7 requires complete reference samples and their actual physical radii.")
    if not np.all(np.isfinite(values)) or not np.allclose(physical_radii, radius[:, None] * xi[None, :], atol=1e-12, rtol=0):
        raise ValueError("The reference field must cover the actual moving domain without exterior samples.")
    meta = solution["metadata"]
    finish = float(meta["finish_time_s"])
    threshold = float(meta.get("threshold", .15))
    if not np.isclose(times[-1], finish, atol=1e-7, rtol=0) or not float(solution["max_C"][-1]) < threshold:
        raise ValueError("Figure 7 needs a completed, strictly dry endpoint.")
    records = _model_series(solution, comparison_solutions)
    comparisons = comparison_solutions or {}
    color_reference = comparisons.get("q3", solution)
    color_values = np.asarray(color_reference["moisture"], dtype=float)
    vmin, vmax = float(np.nanmin(color_values)), float(np.nanmax(color_values))
    low, high = float(values.min()) < vmin, float(values.max()) > vmax
    extend = "both" if low and high else "min" if low else "max" if high else "neither"

    fig, (field, criterion) = plt.subplots(2, 1, figsize=(7.0, 6.1), layout="constrained",
                                         gridspec_kw={"height_ratios": [1, 1]})
    display = np.unique(np.linspace(0, len(times) - 1, min(len(times), 1800)).astype(int))
    shown_h = times[display] / 3600
    shown_r = physical_radii[display].T * 100
    shown_c = values[display].T
    shown_x = np.broadcast_to(shown_h, shown_r.shape)
    # Nodal quadrilaterals end exactly at xi=1; the material exterior is not colored.
    mesh = field.pcolormesh(shown_x, shown_r, shown_c, shading="gouraud", cmap="Blues",
                           vmin=vmin, vmax=vmax, rasterized=True)
    bar = fig.colorbar(mesh, ax=field, pad=.017, aspect=29, extend=extend)
    bar.set_label("干基含水率 / (kg/kg)", fontsize=9)
    bar.ax.tick_params(labelsize=8.5)
    # Clip any display quadrilateral interpolation back to the actual saved surface.
    field.fill_between(times / 3600, radius * 100, 2, color="white", zorder=2)
    field.plot(times / 3600, radius * 100, color="#222222", lw=1.3, zorder=4,
               label=r"实际表面 $r=R(t)$")
    if float(shown_c.min()) < threshold < float(shown_c.max()):
        contour = field.contour(shown_x, shown_r, shown_c, levels=[threshold], colors="black", linewidths=1.15)
        contour.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
        field.clabel(contour, inline=True, fontsize=9, fmt={threshold: f"C={threshold:g}"},
                     manual=[(.66 * finish / 3600, float(radius[0]) * 50)])
    field.scatter([finish / 3600], [float(solution["max_r"][-1]) * 100], marker="*", s=60,
                  color=ORANGE, edgecolor="white", lw=.6, zorder=5, clip_on=False)
    field.set(xlim=(0, finish / 3600), ylim=(0, 2), ylabel="实际半径 r / cm", xlabel="时间 / h")
    field.set_yticks([0, .5, 1, 1.5, 2])
    field.xaxis.set_major_locator(MaxNLocator(7))
    field.set_title("(a) 收缩域含水率", loc="left", fontsize=10)
    field.legend(loc="upper right", frameon=False, fontsize=9)

    finishes = []
    model_notes = []
    for label, result, color, style in records:
        t = np.asarray(result["times"], dtype=float)
        maximum = np.asarray(result["max_C"], dtype=float)
        ending = float(result["metadata"]["finish_time_s"])
        if maximum.shape != t.shape or not np.isfinite(maximum[-1]):
            raise ValueError("Every compared model needs its saved full-grid maximum and endpoint.")
        # Q3 deliberately retains NaNs before 3 h; these are never interpolated away.
        decimals = 3 if result["metadata"].get("shrinking") is False else 4
        criterion.plot(t / 3600, maximum, color=color, ls=style, lw=1.7,
                       label=f"{label}（{ending / 3600:.{decimals}f} h）")
        criterion.scatter([ending / 3600], [maximum[-1]], color=color, edgecolor="white",
                          s=38, lw=.65, zorder=4, marker="o" if style != "-" else "*")
        criterion.plot([ending / 3600] * 2, [0, maximum[-1]], color=color, ls=":", lw=.9)
        finishes.append(ending)
        model_notes.append({"model": label, "finish_time_s": ending,
                            "maximum_first_available_time_s": float(t[np.flatnonzero(np.isfinite(maximum))[0]]),
                            "terminal_max_C": float(maximum[-1])})
    criterion.axhline(threshold, color=RED, ls="--", lw=1.15, label=rf"阈值 ${threshold:g}$ kg/kg")
    criterion.set(xlim=(0, max(finishes) / 3600 * 1.015), ylim=(0, 2.75),
                  xlabel="时间 / h", ylabel="全域最大干基含水率 / (kg/kg)")
    criterion.grid(True)
    criterion.set_title("(b) 全域达标对照", loc="left", fontsize=10)
    criterion.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, Path(folder), "fig7_q4_shrinking_comparison")
    notes = {"figure": "fig7_q4_shrinking_comparison", "models": model_notes,
             "notes": ["各曲线只连接已有全域最大值，问题3前3小时原生最大值未保存，保持NaN空白。",
                       "热图用完整材料参考场映射至实际半径；实际表面之外留白，不填零或表面值。",
                       "三组终点均取保存的未舍入全域严格判据，图例显示值仅用于阅读。",
                       "收缩效应只比较附录4同物性的固定半径组与收缩组。问题3与问题4同时改变物性，不能全归因于收缩。",
                       "图中已移除局部小插图；完整事件轨迹仍保存在原始数值结果。"]}
    (Path(folder) / "fig7_plot_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
