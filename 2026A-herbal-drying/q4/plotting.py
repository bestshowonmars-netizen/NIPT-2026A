"""Q4 input geometry and solved moving-domain figures; no synthetic solution data."""
from pathlib import Path

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
    data = np.asarray(radius_data, dtype=float)
    slopes = np.asarray(radius_slopes, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or slopes.shape != (len(data),):
        raise ValueError("Figure 6 requires radius observations in seconds/metres and their solver PCHIP slopes.")
    fine = np.unique(np.r_[np.linspace(data[0, 0], data[-1, 0], 1801), data[:, 0]])
    hours, radius_cm = fine / 3600, _radius_curve(fine, data, slopes) * 100
    fig = plt.figure(figsize=(6.6, 4.6), layout="constrained")
    layout = fig.add_gridspec(1, 2, width_ratios=[1.03, 1])
    ax = fig.add_subplot(layout[0])
    right = layout[1].subgridspec(2, 1)
    physical = fig.add_subplot(right[0])
    reference = fig.add_subplot(right[1], sharex=physical)

    ax.scatter(data[:, 0] / 3600, data[:, 1] * 100, color=GRAY, s=10,
               alpha=.7, zorder=2, label=f"附件2观测（{len(data)}点）")
    ax.plot(hours, radius_cm, color=GREEN, lw=1.4, label="模型采用的PCHIP插值", zorder=3)
    ax.set(title="(a) 收缩半径输入", xlabel="时间 / h", ylabel="药材半径 / cm")
    ax.set_xlim(hours[0], hours[-1])
    ax.set_ylim(float(data[:, 1].min() * 100) - .055, float(data[:, 1].max() * 100) + .055)
    ax.set_xticks([0, 24, 48, 72])
    ax.grid(True)
    ax.legend(loc="center", bbox_to_anchor=(.62, .37), fontsize=7, frameon=False)
    ax.annotate(f"72 h：{data[-1, 1] * 100:.3f} cm", xy=(hours[-1], radius_cm[-1]),
                xytext=(.49, .25), textcoords="axes fraction", fontsize=7.5,
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": .7})
    early = ax.inset_axes([.42, .52, .55, .41])
    early_mask = fine <= 6 * 3600
    observed_early = data[:, 0] <= 6 * 3600
    early.scatter(data[observed_early, 0] / 3600, data[observed_early, 1] * 100,
                  s=9, color=GRAY, alpha=.75)
    early.plot(hours[early_mask], radius_cm[early_mask], color=GREEN, lw=1.1)
    early.set_xlim(0, 6)
    early.set_title("前6小时", fontsize=7.5, pad=3)
    early.set_xlabel("时间 / h", fontsize=7, labelpad=1)
    early.set_ylabel("半径 / cm", fontsize=7, labelpad=1)
    early.set_xticks([0, 3, 6])
    early.tick_params(labelsize=6.5, pad=1)
    early.grid(True, alpha=.7)

    physical.fill_between(hours, 0, radius_cm, color="#E7EEF4")
    physical.plot(hours, radius_cm, color=GREEN, lw=1.4)
    for xi in (.25, .5, .75):
        physical.plot(hours, radius_cm * xi, color=BLUE, lw=.75, ls=":")
    physical.set(title="(b) 移动物理区域", ylabel="实际半径 r / cm", ylim=(0, 2.05))
    physical.text(.53, .91, r"$r=R(t)$", transform=physical.transAxes, color=GREEN, fontsize=8)
    physical.text(.48, .41, r"$r=\xi R(t)$", transform=physical.transAxes, fontsize=9,
                  bbox={"facecolor": "#E7EEF4", "alpha": .9, "edgecolor": "none", "pad": 1})
    physical.tick_params(labelbottom=False)
    reference.fill_between(hours, 0, 1, color="#E7EEF4")
    for xi in (.25, .5, .75):
        reference.axhline(xi, color=BLUE, lw=.75, ls=":")
    reference.axhline(1, color=GREEN, lw=1.4)
    reference.set(title="(c) 固定参考区域", xlabel="时间 / h", ylabel=r"归一化半径 $\xi=r/R(t)$",
                  xlim=(hours[0], hours[-1]), ylim=(0, 1.03))
    reference.text(.51, .44, "同比径向收缩\n固定ξ随材料运动", transform=reference.transAxes,
                   ha="center", fontsize=7.5, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    for current in (physical, reference):
        current.set_xticks([0, 24, 48, 72])
        current.grid(True, alpha=.7)
    save(fig, folder, "fig6_q4_radius_mapping")


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

    fig, (field, criterion) = plt.subplots(2, 1, figsize=(6.6, 6.4), layout="constrained",
                                         gridspec_kw={"height_ratios": [1, 1.1]})
    display = np.unique(np.linspace(0, len(times) - 1, min(len(times), 1800)).astype(int))
    shown_h = times[display] / 3600
    shown_r = physical_radii[display].T * 100
    shown_c = values[display].T
    shown_x = np.broadcast_to(shown_h, shown_r.shape)
    # Nodal quadrilaterals end exactly at xi=1; the material exterior is not colored.
    mesh = field.pcolormesh(shown_x, shown_r, shown_c, shading="gouraud", cmap="Blues",
                           vmin=vmin, vmax=vmax, rasterized=True)
    bar = fig.colorbar(mesh, ax=field, pad=.017, aspect=29, extend=extend)
    bar.set_label("干基含水率 / (kg/kg)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    field.fill_between(times / 3600, radius * 100, 2, color="#F2F3F4", zorder=2)
    field.plot(times / 3600, radius * 100, color="#222222", lw=1.3, zorder=4,
               label=r"实际表面 $r=R(t)$")
    field.text(.58, .88, "材料域外", transform=field.transAxes, color="#727A80", fontsize=8,
               bbox={"facecolor": "#F2F3F4", "edgecolor": "none", "pad": 1})
    if float(shown_c.min()) < threshold < float(shown_c.max()):
        contour = field.contour(shown_x, shown_r, shown_c, levels=[threshold], colors="black", linewidths=1.15)
        contour.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
        field.clabel(contour, inline=True, fontsize=7.5, fmt={threshold: "C=0.15"})
    field.scatter([finish / 3600], [float(solution["max_r"][-1]) * 100], marker="*", s=60,
                  color=ORANGE, edgecolor="white", lw=.6, zorder=5, clip_on=False)
    field.set(xlim=(0, finish / 3600), ylim=(0, 2), ylabel="实际半径 r / cm", xlabel="从题设初态起的时间 / h")
    field.set_yticks([0, .5, 1, 1.5, 2])
    field.xaxis.set_major_locator(MaxNLocator(7))
    field.set_title(f"(a) 移动区域内的含水率场；终点 {finish / 3600:.4f} h", loc="left", fontsize=9)
    field.legend(loc="upper right", frameon=False, fontsize=7.5)

    finishes = []
    for label, result, color, style in records:
        t = np.asarray(result["times"], dtype=float)
        maximum = np.asarray(result["max_C"], dtype=float)
        ending = float(result["metadata"]["finish_time_s"])
        if maximum.shape != t.shape or not np.isfinite(maximum[-1]):
            raise ValueError("Every compared model needs its saved full-grid maximum and endpoint.")
        # Q3 deliberately retains NaNs before 3 h; these are never interpolated away.
        decimals = 3 if result["metadata"].get("shrinking") is False else 4
        criterion.plot(t / 3600, maximum, color=color, ls=style, lw=1.4,
                       label=f"{label}（{ending / 3600:.{decimals}f} h）")
        criterion.scatter([ending / 3600], [maximum[-1]], color=color, edgecolor="white",
                          s=29, lw=.65, zorder=4, marker="o" if style != "-" else "*")
        finishes.append(ending)
    criterion.axhline(threshold, color=RED, ls="--", lw=1, label="严格达标：M < 0.15 kg/kg")
    criterion.set(xlim=(0, max(finishes) / 3600 * 1.015), ylim=(0, 2.75),
                  xlabel="从题设初态起的时间 / h", ylabel="全域最大干基含水率 / (kg/kg)")
    criterion.grid(True)
    criterion.set_title("(b) 三组模型的全域干燥判据" if len(records) == 3 else "(b) 已有模型的全域干燥判据",
                        loc="left", fontsize=9)
    criterion.legend(loc="upper right", frameon=False, fontsize=7)
    if comparisons.get("q3") is not None:
        criterion.text(.015, .96, "问题3：展示3 h后的续算结果", transform=criterion.transAxes,
                       fontsize=7, color=GRAY, va="top")
    ordered_finishes = sorted(finishes)
    if len(finishes) >= 3 and max(finishes) > 1.5 * min(finishes):
        split = int(np.argmax(np.diff(ordered_finishes))) + 1
        windows = [([.40, .36, .25, .32], ordered_finishes[:split], "较早终点附近"),
                   ([.71, .36, .25, .32], ordered_finishes[split:], "较晚终点附近")]
    else:
        windows = [([.50, .28, .47, .38], ordered_finishes, "阈值附近：实际时间")]
    for rectangle, local_finishes, title in windows:
        inset = criterion.inset_axes(rectangle)
        inset.axhline(threshold, color=RED, lw=.8, ls="--")
        for _, result, color, style in records:
            inset.plot(np.asarray(result["times"]) / 3600, result["max_C"], color=color, ls=style, lw=1)
            ending = float(result["metadata"]["finish_time_s"])
            inset.scatter([ending / 3600], [result["max_C"][-1]], s=16, color=color, zorder=4)
            trial = np.asarray(result.get("event_trace", []), dtype=float)
            if trial.ndim == 2 and trial.shape[1] == 4:
                inset.scatter(trial[:, 0] / 3600, trial[:, 1], s=4, color=color, alpha=.5, zorder=3)
        inset.set_xlim(min(local_finishes) / 3600 * .96, max(local_finishes) / 3600 * 1.01)
        inset.set_ylim(.145, .19)
        inset.set_title(title, fontsize=7.2, pad=3)
        inset.set_xlabel("时间 / h", fontsize=6.7, labelpad=1)
        inset.set_ylabel("最大含水率", fontsize=6.7, labelpad=1)
        inset.tick_params(labelsize=6.3, pad=1)
        inset.xaxis.set_major_locator(MaxNLocator(3))
        inset.yaxis.set_major_locator(MaxNLocator(3))
        inset.grid(True, alpha=.7)
    save(fig, Path(folder), "fig7_q4_shrinking_comparison")
