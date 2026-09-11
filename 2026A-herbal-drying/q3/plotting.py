"""Figure 5: the computed stopping event and the full moisture history."""
from pathlib import Path

import numpy as np
from common.plotting import BLUE, ORANGE, GRAY, setup, save

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator, ScalarFormatter

def figure5(solution, folder):
    """Plot saved solver states and event trials without locating a new root."""
    setup()
    times = np.asarray(solution["times"], dtype=float)
    radii = np.asarray(solution["radii"], dtype=float)
    moisture = np.asarray(solution["moisture"], dtype=float)
    maxima = np.asarray(solution["max_C"], dtype=float)
    max_r = np.asarray(solution["max_r"], dtype=float)
    meta = solution["metadata"]
    threshold = float(meta.get("threshold", 0.15))
    finish = float(meta["finish_time_s"])
    hours = times / 3600.0
    finite = np.isfinite(maxima) & np.isfinite(max_r)
    if not finite[-1] or not np.isclose(times[-1], finish, rtol=0, atol=1e-7):
        raise ValueError("Figure 5 requires the saved global maximum at the exact endpoint.")

    fig, (ax, field) = plt.subplots(
        2, 1, figsize=(6.5, 6.0), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    first_valid = int(np.flatnonzero(finite)[0])
    if first_valid:
        ax.axvspan(hours[0], hours[first_valid], color="#F1F2F3", zorder=0)
        ax.text(.015, .975, "0–3 h：见问题2", transform=ax.transAxes,
                fontsize=7.5, va="top", color="#596168",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    # The NaNs in the old Q2 history deliberately leave the global-max curve blank.
    ax.plot(hours, maxima, color=BLUE, lw=1.5, label=r"全域最大值 $C_{\max}(t)$")
    ax.axhline(threshold, color="#161616", lw=1.1, ls="--",
               label=r"终点阈值 $0.15$ kg/kg")
    ax.scatter([finish / 3600], [maxima[-1]], s=27, facecolor="white",
               edgecolor="#111111", zorder=6, clip_on=False)
    ax.annotate(
        f"终点 {finish / 3600:.4f} h\n最大值位置 r = {max_r[-1] * 100:.4f} cm",
        xy=(finish / 3600, maxima[-1]), xycoords="data",
        xytext=(.62, .25), textcoords="axes fraction", fontsize=8,
        ha="center", va="bottom",
        arrowprops={"arrowstyle": "->", "color": "#454545", "lw": .8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2},
    )
    ax.set_ylabel(r"最大干基含水率 / (kg/kg)")
    ax.set_ylim(0, max(float(np.nanmax(maxima)) * 1.09, threshold * 1.5))
    ax.grid(True)
    ax.text(.012, 1.02, "(a) 全域阈值判定", transform=ax.transAxes, fontsize=9, va="bottom")
    ax.legend(loc="center left", bbox_to_anchor=(.14, .66), frameon=False, fontsize=8)

    trace = np.asarray(solution["event_trace"], dtype=float)
    if trace.ndim != 2 or trace.shape[1] != 4 or not len(trace):
        raise ValueError("Figure 5 needs actual [t, M, r_max, g] event trials.")
    bracket = meta["bracket"]
    width = float(bracket["width"])
    local_span = max(2.0 * float(meta.get("event_dt", 1.0)), 16.0 * width, 1.0)
    chosen = np.abs(trace[:, 0] - finish) <= local_span
    if np.count_nonzero(chosen) < 3:
        chosen[np.argsort(np.abs(trace[:, 0] - finish))[:min(6, len(trace))]] = True
    local = trace[chosen]
    inset = ax.inset_axes([.52, .56, .44, .37])
    inset.axhline(0, color="#161616", ls="--", lw=.9)
    inset.axvspan(float(bracket["t_minus"]) - finish,
                  float(bracket["t_plus"]) - finish,
                  facecolor="#E2E8ED", edgecolor="none", zorder=0)
    # Trial points can follow different integration paths: do not spline or connect them.
    inset.scatter(local[:, 0] - finish, local[:, 3], s=13,
                  facecolor="white", edgecolor=BLUE, lw=.7, zorder=3)
    inset.scatter([0], [maxima[-1] - threshold], s=19,
                  facecolor="#111111", edgecolor="white", lw=.45, zorder=4)
    inset.set_title("交点附近：实际积分试算", fontsize=7.5, pad=2)
    inset.set_xlabel(r"$t-t_+$ / s", fontsize=7, labelpad=1)
    inset.set_ylabel(r"$C_{\max}-0.15$", fontsize=7, labelpad=1)
    inset.tick_params(labelsize=6.5, pad=1)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    inset.yaxis.set_major_formatter(formatter)
    inset.yaxis.get_offset_text().set_fontsize(6.5)
    inset.grid(True, alpha=.65)
    inset.margins(x=.12, y=.2)

    # Display subsampling bounds file size; every displayed value is a saved state.
    display = np.unique(np.linspace(0, len(times) - 1, min(len(times), 2000)).astype(int))
    shown_h = hours[display]
    shown_c = moisture[display]
    image = field.pcolormesh(shown_h, radii * 100, shown_c.T,
                            cmap="Blues", shading="auto", rasterized=True)
    bar = fig.colorbar(image, ax=field, pad=.018, aspect=27)
    bar.set_label("干基含水率 / (kg/kg)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    available = [v for v in [.3, .6, 1.2]
                 if float(shown_c.min()) < v < float(shown_c.max())]
    if available:
        ordinary = field.contour(shown_h, radii * 100, shown_c.T,
                                 levels=available, colors="#6D7A83", linewidths=.55, alpha=.7)
        field.clabel(ordinary, inline=True, fontsize=6.5, fmt="%g")
    if float(shown_c.min()) < threshold < float(shown_c.max()):
        stopping = field.contour(shown_h, radii * 100, shown_c.T,
                                  levels=[threshold], colors="black", linewidths=1.15)
        stopping.set_path_effects([pe.Stroke(linewidth=2.7, foreground="white"), pe.Normal()])
        field.clabel(stopping, inline=True, fontsize=8, fmt={threshold: "C=0.15"})
    field.plot(hours, max_r * 100, color=ORANGE, ls=":", lw=1.0, zorder=4, clip_on=False,
               label=r"全域最大值位置 $r_{\max}$",
               path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    field.scatter([finish / 3600], [max_r[-1] * 100], marker="*", s=54,
                  color=ORANGE, edgecolor="white", lw=.6, zorder=5, clip_on=False)
    field.text(.012, 1.02, "(b) 全过程含水率场", transform=field.transAxes, fontsize=9, va="bottom")
    field.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=7.5)
    field.set_ylabel("到中心的距离 / cm")
    field.set_xlabel("从题设初态起的时间 / h")
    field.set_ylim(float(radii[0] * 100), float(radii[-1] * 100))
    field.set_xlim(0, finish / 3600)
    field.set_yticks([0, .5, 1, 1.5, 2])
    field.xaxis.set_major_locator(MaxNLocator(7))
    save(fig, Path(folder), "fig5_q3_drying_endpoint")
