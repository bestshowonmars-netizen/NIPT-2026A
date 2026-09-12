"""Figure 5: the computed stopping event and the full moisture history."""
from pathlib import Path
import json

import numpy as np
from common.plotting import BLUE, ORANGE, GRAY, setup, save

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

RED = "#B64545"


def _mean_threshold_bracket(times, means, threshold):
    """Return adjacent saved whole-minute states, never an interpolated root."""
    regular = np.flatnonzero(np.abs(times / 60.0 - np.rint(times / 60.0)) < 1e-9)
    below = np.flatnonzero(means[regular] < threshold)
    if not len(below):
        return None
    first = int(below[0])
    if first == 0:
        raise ValueError("The saved initial mean is already below the requested threshold.")
    left, right = int(regular[first - 1]), int(regular[first])
    if not (means[left] >= threshold and times[right] - times[left] == 60.0):
        raise ValueError("Mean threshold display requires adjacent whole-minute saved states.")
    return {"t_minus_s": float(times[left]), "t_plus_s": float(times[right]),
            "mean_minus": float(means[left]), "mean_plus": float(means[right]),
            "interpretation": "Saved 60-second bracket only; no precise mean-threshold root was computed."}

def figure5(solution, folder):
    """Compare maximum and mean on an hour scale using only saved states."""
    setup()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    times = np.asarray(solution["times"], dtype=float)
    radii = np.asarray(solution["radii"], dtype=float)
    moisture = np.asarray(solution["moisture"], dtype=float)
    maxima = np.asarray(solution["max_C"], dtype=float)
    means = np.asarray(solution["mean_C"], dtype=float)
    max_r = np.asarray(solution["max_r"], dtype=float)
    meta = solution["metadata"]
    threshold = float(meta.get("threshold", 0.15))
    finish = float(meta["finish_time_s"])
    hours = times / 3600.0
    finite = np.isfinite(maxima) & np.isfinite(max_r)
    if not finite[-1] or not np.isclose(times[-1], finish, rtol=0, atol=1e-7):
        raise ValueError("Figure 5 requires the saved global maximum at the exact endpoint.")
    if means.shape != times.shape or not np.all(np.isfinite(means)):
        raise ValueError("Figure 5 requires the saved dry-mass-weighted mean throughout the run.")
    mean_bracket = _mean_threshold_bracket(times, means, threshold)

    fig, (ax, field) = plt.subplots(
        2, 1, figsize=(7.0, 6.2), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    # The NaNs in the old Q2 history deliberately leave the global-max curve blank.
    ax.plot(hours, maxima, color=BLUE, lw=1.7, label=r"全域最大值 $M(t)$")
    ax.plot(hours, means, color=ORANGE, lw=1.6, ls="-.", label=r"干质量加权均值 $\overline{C}(t)$")
    ax.axhline(threshold, color=RED, lw=1.15, ls="--",
               label=rf"阈值 ${threshold:g}$ kg/kg")
    ax.scatter([finish / 3600], [maxima[-1]], s=27, facecolor="white",
               edgecolor="#111111", zorder=6, clip_on=False)
    ax.annotate(
        f"全域达标 {finish / 3600:.4f} h\n最湿位置 r = {max_r[-1] * 100:.4f} cm",
        xy=(finish / 3600, maxima[-1]), xycoords="data",
        xytext=(.76, .31), textcoords="axes fraction", fontsize=9,
        ha="center", va="bottom",
        arrowprops={"arrowstyle": "->", "color": "#454545", "lw": .8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2},
    )
    if mean_bracket is not None:
        lower_h, upper_h = mean_bracket["t_minus_s"] / 3600, mean_bracket["t_plus_s"] / 3600
        ax.axvspan(lower_h, upper_h, color=ORANGE, alpha=.18, zorder=0)
        ax.scatter([lower_h, upper_h], [mean_bracket["mean_minus"], mean_bracket["mean_plus"]],
                   marker="s", s=22, facecolor="white", edgecolor=ORANGE, lw=.9, zorder=5)
        ax.annotate(f"均值跨阈值区间\n[{lower_h:.4f}, {upper_h:.4f}] h",
                    xy=(upper_h, mean_bracket["mean_plus"]), xycoords="data",
                    xytext=(.41, .49), textcoords="axes fraction", ha="center", fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": .8},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    ax.set_ylabel(r"干基含水率 / (kg/kg)")
    ax.set_ylim(0, max(float(np.nanmax(maxima)), float(means.max())) * 1.08)
    ax.grid(True)
    ax.set_title("(a) 最大值与平均值", loc="left", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Display subsampling bounds file size; every displayed value is a saved state.
    display = np.unique(np.linspace(0, len(times) - 1, min(len(times), 2000)).astype(int))
    shown_h = hours[display]
    shown_c = moisture[display]
    image = field.pcolormesh(shown_h, radii * 100, shown_c.T,
                            cmap="Blues", shading="auto", rasterized=True)
    bar = fig.colorbar(image, ax=field, pad=.018, aspect=27)
    bar.set_label("干基含水率 / (kg/kg)", fontsize=9)
    bar.ax.tick_params(labelsize=8.5)
    available = [v for v in [.3, .6, 1.2]
                 if float(shown_c.min()) < v < float(shown_c.max())]
    if available:
        ordinary = field.contour(shown_h, radii * 100, shown_c.T,
                                 levels=available, colors="#6D7A83", linewidths=.55, alpha=.7)
        for level, path in zip(ordinary.levels, ordinary.get_paths()):
            vertices = path.vertices
            if len(vertices):
                center_height = (radii[0] + radii[-1]) * 50
                label_point = vertices[np.argmin(np.abs(vertices[:, 1] - center_height))]
                field.clabel(ordinary, levels=[level], inline=True, fontsize=8, fmt="%g",
                             manual=[tuple(label_point)])
    if float(shown_c.min()) < threshold < float(shown_c.max()):
        stopping = field.contour(shown_h, radii * 100, shown_c.T,
                                  levels=[threshold], colors="black", linewidths=1.15)
        stopping.set_path_effects([pe.Stroke(linewidth=2.7, foreground="white"), pe.Normal()])
        field.clabel(stopping, inline=True, fontsize=9, fmt={threshold: f"C={threshold:g}"})
    field.plot(hours, max_r * 100, color=ORANGE, ls=":", lw=1.0, zorder=4, clip_on=False,
               label=r"全域最大值位置 $r_{\max}$",
               path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    field.scatter([finish / 3600], [max_r[-1] * 100], marker="*", s=54,
                  color=ORANGE, edgecolor="white", lw=.6, zorder=5, clip_on=False)
    field.set_title("(b) 含水率场", loc="left", fontsize=10)
    field.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8.5)
    field.set_ylabel("到中心的距离 / cm")
    field.set_xlabel("从题设初态起的时间 / h")
    field.set_ylim(float(radii[0] * 100), float(radii[-1] * 100))
    field.set_xlim(0, finish / 3600)
    field.set_yticks([0, .5, 1, 1.5, 2])
    field.xaxis.set_major_locator(MaxNLocator(7))
    save(fig, Path(folder), "fig5_q3_drying_endpoint")
    notes = {"figure": "fig5_q3_drying_endpoint", "finish_time_s": finish,
             "maximum_first_available_time_s": float(times[np.flatnonzero(finite)[0]]),
             "mean_threshold_saved_minute_bracket": mean_bracket,
             "notes": ["前3小时未保存原生网格最大值；最大值及其位置曲线保持为空，不以采样最大值补造。",
                       "平均值低于阈值不能证明所有位置达标；全域严格达标只使用未舍入的最大值。",
                       "均值标注仅为相邻60秒保存时刻的阈值夹区，未求解或插值生成精确均值达标时刻。",
                       "事件试算保留在原始结果event_trace，主图使用小时尺度，不嵌入10^-7量级含水率差的小插图。"]}
    (Path(folder) / "fig5_plot_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
