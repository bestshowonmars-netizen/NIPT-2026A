"""Render Q3 verification evidence already on disk; never launch a solver."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
from validation.evidence_paths import project_path

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math

import numpy as np
from common.plotting import setup, save, BLUE, ORANGE, GRAY
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = bootstrap.ROOT
MATRICES = ("time", "space", "event", "event_step", "picard")
KINDS = {"time": "时间步", "space": "空间网格", "space_cross_two_levels": "空间网格跨两档",
         "q2_seed_sensitivity": "Q2末态来源",
         "event_tolerance": "事件夹逼容差", "event_integration_step": "事件积分步长",
         "picard_tolerance": "Picard容差"}


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def _path(value, root):
    return project_path(value, root)


def load_evidence(root=ROOT):
    """Collect serialized comparisons and their run metadata, not field arrays."""
    root = Path(root)
    reports, runs, comparisons = {}, {}, []
    for name in MATRICES:
        data = _read_json(root / "validation" / f"q3_convergence_{name}.json")
        if data is None:
            continue
        reports[name] = data
        for original in data.get("runs", []):
            run = dict(original)
            meta = _read_json(_path(run["path"], root).with_suffix(".json")) or {}
            run["metadata"] = meta
            # Local event checks intentionally serialize only a small run summary.
            # Actual numerical settings come from the persisted result metadata.
            for key, meta_key in (("cells", "cells"), ("dt", "dt_max"),
                                  ("event_dt", "event_dt"), ("event_tol", "epsilon_t"),
                                  ("finish_time_s", "finish_time_s"), ("source_path", "source_path")):
                if key in run:
                    run[key + "_requested"] = run[key]
                if meta_key in meta:
                    run[key] = meta[meta_key]
            if "finish_time_s" not in run:
                run["finish_time_s"] = run.get("checks", {})["finish_time_s"]
            if "source" not in run:
                native = _path(run["source_path"], root).resolve()
                run["source"] = "production" if native == (root / "q2/solution.npz").resolve() else "matched_q2"
            run.setdefault("tight", False)
            runs[run["label"]] = run
        for comparison in data.get("comparisons", []):
            comparisons.append({**comparison, "report_matrix": name})
    ordered = sorted(runs.values(), key=lambda r: (r.get("source", ""), int(r["cells"]),
                     -float(r["dt"]), -float(r["event_dt"]), -float(r["event_tol"]),
                     bool(r.get("tight", False)), r["label"]))
    labels = {r["label"]: f"R{i + 1:02d}" for i, r in enumerate(ordered)}
    return {"root": root, "reports": reports, "runs": runs, "ordered_runs": ordered,
            "comparisons": comparisons, "labels": labels,
            "selected": _read_json(root / "validation/q3_selected_config.json"),
            "acceptance": _read_json(root / "validation/q3_acceptance.json"),
            "reproduction": _read_json(root / "validation/q3_reproduction.json"),
            "prefix_reproduction": _read_json(root / "paper_appendix_minimal/reproduced/q3_prefix/prefix_comparison.json"),
            "boundary_sensitivity": _read_json(root / "validation/q3_boundary_sensitivity.json"),
            "core": _read_json(root / "validation/q3_core_checks.json"),
            "formal_checks": _read_json(root / "validation/q3_checks.json")}


def _seed_scale(run):
    return run.get("metadata", {}).get("source_metadata", {}).get("dt_scale")


def _fraction(value):
    if value is None:
        return "未记录"
    ratio = Fraction(float(value)).limit_denominator(100000)
    return str(ratio) if ratio.denominator != 1 else str(ratio.numerator)


def _selected_fields(selected):
    if not isinstance(selected, dict):
        return {}
    fields = dict(selected)
    for key in ("config", "configuration", "selected", "selected_run"):
        if isinstance(selected.get(key), dict):
            fields.update(selected[key])
    return fields


def _selected_run(evidence):
    fields = _selected_fields(evidence["selected"])
    label = next((fields[k] for k in ("selected_run_label", "selected_label", "run_label", "label")
                  if isinstance(fields.get(k), str)), None)
    if label in evidence["runs"]:
        return evidence["runs"][label]
    path = next((fields[k] for k in ("selected_run_path", "run_path", "path", "solution_path")
                 if isinstance(fields.get(k), str)), None)
    if path:
        target = _path(path, evidence["root"]).resolve()
        for run in evidence["ordered_runs"]:
            if _path(run["path"], evidence["root"]).resolve() == target:
                return run
    return None


def _panel_empty(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color=GRAY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def figure8(evidence, folder):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), layout="constrained")
    all_finishes = [float(r["finish_time_s"]) for r in evidence["ordered_runs"]]
    origin = math.floor(min(all_finishes) / 10) * 10 if all_finishes else 0
    fig.suptitle(f"终点时间减去 {origin:g} s 后显示；该平移量不是真解或误差基准", fontsize=8)
    colors = [BLUE, ORANGE, "#547B68", "#71577D"]
    selected = _selected_run(evidence)

    time_runs = [r for r in evidence["reports"].get("time", {}).get("runs", [])
                 if r.get("source") == "production"]
    time_groups = defaultdict(list)
    for original in time_runs:
        r = evidence["runs"][original["label"]]
        time_groups[(int(r["cells"]), float(r["event_dt"]), float(r["event_tol"]),
                     bool(r.get("tight", False)), _seed_scale(r))].append(r)
    for number, (group, values) in enumerate(sorted(time_groups.items(), key=lambda item: str(item[0]))):
        values.sort(key=lambda r: float(r["dt"]))
        label = f"N={group[0]}，同一Q2末态" if len(time_groups) == 1 else f"事件步长 {group[1]:g} s，ε={group[2]:g}"
        axes[0].plot([r["dt"] for r in values], [float(r["finish_time_s"]) - origin for r in values],
                     "o-", color=colors[number % len(colors)], ms=4, lw=1.1, label=label)
    if time_runs:
        # A log x-axis shows geometric step refinements without hiding fine settings.
        axes[0].set_xscale("log", base=2)
        ticks = sorted({float(r["dt"]) for r in time_runs})
        axes[0].set_xticks(ticks, [_fraction(v) for v in ticks], rotation=30 if len(ticks) > 6 else 0)
        axes[0].legend(frameon=False, fontsize=7.5, loc="best")
        seed_text = ", ".join(sorted({_fraction(_seed_scale(evidence['runs'][r['label']])) for r in time_runs}))
        cell_text = ", ".join(str(n) for n in sorted({int(r['cells']) for r in time_runs}))
        note = f"Q2早期尺度 τ={seed_text} s"
        if len(time_groups) > 1:
            note = f"N={cell_text}；同一Q2末态\n" + note
        axes[0].text(.03, .97, note, transform=axes[0].transAxes,
                     fontsize=7, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    else:
        _panel_empty(axes[0], "尚无已完成的时间步比较")

    space_originals = evidence["reports"].get("space", {}).get("runs", [])
    space_groups = defaultdict(list)
    production = []
    for original in space_originals:
        r = evidence["runs"][original["label"]]
        if r.get("source") == "matched_q2":
            seed = r["metadata"].get("source_metadata", {})
            space_groups[(float(r["dt"]), float(r["event_dt"]), float(r["event_tol"]),
                          _seed_scale(r), seed.get("time_growth"))].append(r)
        elif r.get("source") == "production":
            production.append(r)
    for number, (group, values) in enumerate(sorted(space_groups.items(), key=lambda item: str(item[0]))):
        values.sort(key=lambda r: int(r["cells"]))
        label = f"各N自有Q2末态；Δt={group[0]:g} s"
        axes[1].plot([r["cells"] for r in values], [float(r["finish_time_s"]) - origin for r in values],
                     "s-", color=colors[number % len(colors)], ms=4, lw=1.1, label=label)
    for number, run in enumerate(production):
        axes[1].scatter([run["cells"]], [float(run["finish_time_s"]) - origin], marker="D", s=42,
                        facecolor="none", edgecolor=ORANGE, linewidth=.9, zorder=5,
                        label="原Q2末态（独立对照）" if number == 0 else None)
    if space_groups:
        ticks = sorted({int(r["cells"]) for group in space_groups.values() for r in group})
        axes[1].set_xticks(ticks)
        axes[1].legend(frameon=False, fontsize=7, loc="best")
        scales = sorted({_fraction(group[3]) for group in space_groups})
        text = "各组固定Q2早期尺度\nτ=" + ", ".join(scales) + " s；未投影初态"
        axes[1].text(.03, .97, text, transform=axes[1].transAxes, fontsize=7, va="top",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    else:
        _panel_empty(axes[1], "尚无已完成的空间网格比较")

    if selected:
        selection_label = "正式选档" if (evidence.get("acceptance") or {}).get("passed") is True else "选档记录（待验收）"
        if any(r["label"] == selected["label"] for r in time_runs):
            axes[0].scatter([selected["dt"]], [float(selected["finish_time_s"]) - origin],
                            marker="*", s=90, color=ORANGE, edgecolor="white", lw=.7, zorder=7,
                            label=selection_label)
            axes[0].legend(frameon=False, fontsize=7, loc="best")
        if any(r["label"] == selected["label"] for r in space_originals):
            axes[1].scatter([selected["cells"]], [float(selected["finish_time_s"]) - origin],
                            marker="*", s=90, color=ORANGE, edgecolor="white", lw=.7, zorder=7,
                            label=selection_label)
            axes[1].legend(frameon=False, fontsize=7, loc="best")
    for ax, title, xlabel in zip(axes, ["(a) 时间步与终点时间", "(b) 网格与终点时间"],
                                ["续算时间步上限 Δt / s", "径向单元数 N"]):
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"(终点时间 − {origin:g} s) / s")
        ax.grid(True, alpha=.8)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.margins(x=.12, y=.22)
    save(fig, folder, "fig8_q3_convergence")


def _selection_text(evidence):
    selected = evidence["selected"]
    if selected is None:
        return ["最终选档文件尚不存在：本报告只汇总当前已保存的记录，不宣布正式配置通过。", ""]
    fields = _selected_fields(selected)
    lines = ["已读取 `validation/q3_selected_config.json`；各项比较结论仍以本报告下表中的实际差值为准。", "",
             "| 选档字段 | 记录值 |", "|---|---|"]
    names = ("status", "validated", "accepted", "selected_run_label", "selected_run_path", "label", "path",
             "cells", "dt_max", "dt", "event_dt", "event_tol", "epsilon_t", "finish_time_s", "finish_time_h")
    for key in names:
        if key in fields and not isinstance(fields[key], (list, dict)):
            lines.append(f"| {key} | {fields[key]} |")
    run = _selected_run(evidence)
    if run:
        lines.extend(["", f"该选档对应本报告算例 {evidence['labels'][run['label']]}；星号表示该记录（不表示真解）。"])
    else:
        lines.extend(["", "选档文件当前未通过标签/路径对应到本批比较算例，图中不猜测或补画选档星号。"])
    return lines + [""]


def _comparison_row(comparison, evidence):
    coarse = evidence["labels"].get(comparison["coarse"], comparison["coarse"])
    fine = evidence["labels"].get(comparison["fine"], comparison["fine"])
    kind = KINDS.get(comparison["kind"], comparison["kind"])
    if comparison.get("unavailable_run") or "C" not in comparison:
        return f"| {kind} | {coarse}→{fine} | — | — | — | — | 缺失算例，未验证 |"
    c, target = comparison["C"], comparison["targets"]
    recorded = "通过" if comparison["passed"] else "未通过"
    return (f"| {kind} | {coarse}→{fine} | {float(c['maximum']):.8e} | "
            f"({float(c['time_s']):g}, {float(c['radius_cm']):g}) | "
            f"{float(comparison['finish_time_s']['difference']):.9g} | "
            f"{float(target['full_output_C']):g} / {float(target['endpoint_s']):g} | {recorded} |")


def _acceptance_text(evidence):
    acceptance = evidence["acceptance"]
    lines = ["## 正式选档的必需比较", ""]
    if not isinstance(acceptance, dict):
        return lines + ["尚无 `q3_acceptance.json`：目前不根据选档文件自行声明的validated字段代替原数组核验。", ""]
    comparisons = acceptance.get("comparisons_recomputed_from_raw_arrays", [])
    if not comparisons:
        return lines + ["选档核验文件中没有可读取的原数组比较列表，暂不宣布正式精度通过。", ""]
    passed = acceptance.get("passed") is True and all(c.get("passed") is True for c in comparisons)
    arrays_match = acceptance.get("formal_arrays_identical_to_selected_run") is True
    hashes = acceptance.get("current_source_and_input_hashes_verified", [])
    lines.extend(["该部分读取独立验收程序重新从原始数组计算的差值，不用历史粗档的全矩阵标志代替选档判断。", "",
                  f"记录结论：所列必需比较{'全部通过' if passed else '尚未全部通过'}；"
                  f"正式数组与选定算例{'逐项一致' if arrays_match else '一致性未确认'}；"
                  f"数值源和输入哈希核对记录共 {len(hashes)} 项。", "",
                  "| 检查 | 比较 | maxΔC/(kg/kg) | 最大差位置(t/s,r/cm) | Δ终点/s | 限值C/终点s | 记录结论 |",
                  "|---|---|---:|---|---:|---|---|"])
    lines.extend(_comparison_row(c, evidence) for c in comparisons)
    lines.extend(["", "这里的通过限定于规定输出位置、比较档次和预定门槛，不转换为对未知精确解的误差上界。", ""])
    return lines


def _boundary_text(evidence):
    data = evidence["boundary_sensitivity"]
    if not isinstance(data, dict):
        return []
    comparison = data["comparison"]
    first = comparison["coarse_config"]
    second = comparison["fine_config"]
    baseline = data["baseline_tail"]
    control = data["control_tail"]
    before = data["before_4h_maximum_difference"]
    finish = comparison["finish_time_s"]
    lines = ["## 环境延拓假设的专门对照", "",
             f"这组比较固定同一N={first['cells']}网格和Δt={float(data['common_step_s']):g} s续算步长，"
             "只改变4小时后的环境延拓输入；它单独研究输入假设敏感性，不能套用离散误差门槛，也不替代正式细步长验证。",
             "", "| 对照方案 | 4小时后温度/°C | 等效水分驱动/(kg/kg) | 终点/s |", "|---|---:|---:|---:|",
             f"| 既定恒值延拓 | {float(baseline['temperature_K']) - 273.15:.3f} | {float(baseline['moisture']):.5f} | {float(finish['coarse']):.9f} |",
             f"| 保持附件末值 | {float(control['temperature_K']) - 273.15:.3f} | {float(control['moisture']):.5f} | {float(finish['fine']):.9f} |", "",
             f"终点有符号变化为 {float(data['signed_finish_change_s']):.6f} s"
             f"（{float(data['signed_finish_change_s']) / 60:.6f} min），"
             f"相对变化为 {float(data['relative_finish_change_percent']):.6f}%。"
             f"前4小时最大温度差为 {float(before['temperature_K']):.3e} K、最大含水率差为 {float(before['moisture']):.3e} kg/kg；"
             f"对照运行的最大水量残差为 {float(data['control_metadata']['max_water_balance_residual']):.3e} kg/kg。", "",
             f"表中两个终点均属于Δt={float(data['common_step_s']):g} s的专门对照，不是正式选档终点。本结果说明某一种合理延拓替代会改变预测；它不是实验验证、不是参数校准，也不据此更换已约定的正式边界输入。", ""]
    if first["cells"] != second["cells"] or float(first["dt_max"]) != float(second["dt_max"]):
        raise ValueError("The boundary sensitivity comparison does not use matched discretization.")
    return lines


def _array_comparison_rows(records, prefix=""):
    rows = []
    names = {"temperature_K": "温度 / K", "moisture": "含水率 / (kg/kg)",
             "final_T": "温度 / K", "final_C": "含水率 / (kg/kg)",
             "faces": "单元面半径 / m", "centers": "单元中心半径 / m",
             "volumes": "径向体积权重 / m²", "max_C": "最大含水率 / (kg/kg)",
             "max_r": "最大值位置 / m", "main": "主程序", "independent": "附录程序"}
    for key, record in records.items():
        shape = " × ".join(str(n) for n in record.get("shape", [])) or "未记录"
        error = record.get("max_abs_difference")
        error_text = f"{float(error):.8e}" if error is not None else "未记录"
        equality = record.get("array_equal")
        equal = "是" if equality is True else "否" if equality is False else "未记录"
        rows.append(f"| {prefix}{names.get(key, key)} | {shape} | {error_text} | {equal} |")
    return rows


def _reproduction_text(evidence):
    """Separate completed fresh-prefix evidence from the full Q3 comparison."""
    full_completed = isinstance(evidence.get("reproduction"), dict) and evidence["reproduction"].get("passed") is True
    lines = ["## 附录程序的独立运行复现", "",
             "该项核对同一离散方法和算法在自包含附录程序中的独立运行结果，来源链为原始附件与初始状态 → 独立重算Q2前3小时 → 使用该原网格末态续算Q3 → 导出独立结果表。它检验代码与结果可复现性，不属于独立物理模型或内部实测场验证。", ""]
    prefix = evidence.get("prefix_reproduction")
    if isinstance(prefix, dict):
        fresh = prefix.get("from_original_attachment_and_initial_state") is True
        no_cache = prefix.get("main_cache_used_by_independent_solver") is False
        completed = prefix.get("fresh_solve_process_exit_code") == 0
        meta = prefix.get("fresh_metadata", {})
        lines.extend([f"前段来源记录：从原附件和初值重新求解{'已确认' if fresh else '未确认'}；"
                      f"独立求解未使用主程序缓存{'已确认' if no_cache else '未确认'}；"
                      f"该阶段进程{'正常完成' if completed else '完成状态未确认'}。"
                      f"记录的前段终止时间为 {meta.get('end_time', '未记录')} s，径向单元数为 {meta.get('cells', '未记录')}。", "",
                      "| Q2前段核对量 | 数组形状 | 最大绝对差 | 逐项完全相等 |",
                      "|---|---|---:|---|"])
        lines.extend(_array_comparison_rows(prefix.get("full_grid_comparison", {}), "原网格 "))
        lines.extend(_array_comparison_rows(prefix.get("history_21_points_comparison", {}), "逐秒21点 "))
        prefix_conclusion = "此前段证据与下述全程Q3及Excel复核共同构成独立运行来源链。" if full_completed else "这一阶段完成仅证明Q2前段已独立复算，不能代替全程Q3和Excel核对。"
        lines.extend(["", f"前段时间数组相等记录为 {prefix.get('times_equal', '未记录')}，"
                      f"诊断日志相等记录为 {prefix.get('logs_equal', '未记录')}。{prefix_conclusion}", ""])
    else:
        lines.extend(["独立Q2前段的来源与比较记录尚不存在，暂不确认初态来源链。", ""])
    data = evidence.get("reproduction")
    if not isinstance(data, dict):
        return lines + ["尚无 `validation/q3_reproduction.json`：全程Q3独立运行及其终点、结果表核对尚无完成证据，本报告不将其计为通过。", ""]
    config = data.get("configuration", {})
    lines.extend([f"全程复现核对文件的结论为：{'通过' if data.get('passed') is True else '尚未通过'}。"
                  f"配置为 N={config.get('cells', '未记录')}，续算步长上限 {config.get('dt_max', '未记录')} s，"
                  f"事件步长 {config.get('event_dt', '未记录')} s，夹逼容差 {config.get('epsilon_t', '未记录')} s。", "",
                  f"共同整60秒、21个径向位置的每个场核对 {data.get('regular_values_checked_per_field', '未记录')} 个值，"
                  f"时间范围为 {data.get('regular_time_range_s', '未记录')} s；精确终点另列入完整输出核对。", "",
                  "| Q3核对量 | 数组形状 | 最大绝对差 | 逐项完全相等 |", "|---|---|---:|---|"])
    for key, label in (("all_regular_60_second_21_point_fields", "整分21点 "),
                       ("all_outputs_including_exact_endpoint", "含精确终点 "),
                       ("full_960_cell_terminal_state_and_geometry", "原网格末态/几何 "),
                       ("full_grid_maximum_history_after_3h", "3小时后全域最大值 ")):
        lines.extend(_array_comparison_rows(data.get(key, {}), label))
    endpoint = data.get("endpoint", {})
    if endpoint:
        lines.extend(["", f"主程序终点为 {float(endpoint['main_time_s']):.12f} s，"
                      f"附录终点为 {float(endpoint['independent_time_s']):.12f} s，"
                      f"差值为 {float(endpoint['time_difference_s']):.8e} s。"
                      f"附录终点未经舍入的全域最大含水率为 {float(endpoint['strict_independent_final_max_C']):.17g} kg/kg。", "",
                      "| 终点夹逼核对量 | 绝对差 |", "|---|---:|"])
        lines.extend(f"| {key} | {float(value):.8e} |"
                     for key, value in endpoint.get("bracket_abs_differences", {}).items())
        if endpoint.get("event_trace"):
            lines.extend(["", "| 事件试算核对量 | 数组形状 | 最大绝对差 | 逐项完全相等 |", "|---|---|---:|---|",
                          *_array_comparison_rows({"event_trace": endpoint["event_trace"]})])
    excel = data.get("excel", {})
    if excel:
        between = excel.get("between_workbooks", {})
        lines.extend(["", f"两份Excel各核对 {excel.get('data_rows_checked', '未记录')} 个整分数据行、"
                      f"{excel.get('all_C_values_checked_per_workbook', '未记录')} 个含水率数值；"
                      f"两表最大差为 {float(between['max_abs_difference']):.8e}。"
                      "两表还分别与各自未经舍入的NPZ按四位小数导出规则核对。精确的非整分终点未冒充整分写入Excel。", "",
                      "| Excel与自身NPZ核对 | 数组形状 | 最大绝对差 | 逐项完全相等 |", "|---|---|---:|---|"])
        lines.extend(_array_comparison_rows(excel.get("each_workbook_against_own_unrounded_npz", {})))
    water = data.get("water_balance", {})
    if water:
        lines.extend(["", f"附录完整Q2+Q3轨迹的最大水量残差为 {float(water['independent_max_abs_residual']):.8e} kg/kg；"
                      f"两套运行水量残差历史的最大差为 {float(water['between_runs']['max_abs_difference']):.8e} kg/kg。"])
    source = data.get("fresh_q2_source", {})
    if source:
        if source.get("standalone_from_initial") is False:
            lines.extend(["", "本次采用先独立重算Q2、再由该末态启动Q3的两阶段运行；Q3入口的from_initial=false标记按实际保留，来源凭据将两阶段衔接起来。"])
        lines.extend(["", f"全程记录的独立前段来源为 `{source.get('prefix_path', '未记录')}`，"
                      f"其SHA256为 `{source.get('prefix_sha256', '未记录')}`。"
                      f"续算复用了该独立重算前段的确认值为 {source.get('standalone_continuation_reused_independently_computed_prefix', '未记录')}；"
                      f"核对文件共保存 {len(data.get('files_sha256', {}))} 项文件哈希，用于追踪附件、源码、原数组及结果表。"])
    return lines + [""]


def render_markdown(evidence, figure_path):
    missing = [name for name in MATRICES if name not in evidence["reports"]]
    lines = ["# 问题3精度验证报告", "",
             "本报告只读取已保存的验证JSON和算例元数据，没有重新运行数值模拟。缺失记录表示尚无证据，不计为通过；保留粗档失败记录，也不把粗档失败等同于之后细档一定失败。", "",
             "## 当前选档状态", "", *_selection_text(evidence),
             *_acceptance_text(evidence),
             "## 比较量与验收范围", "",
             r"令 $\mathcal S_{ab}$ 为两组计算共同存在的整60秒、21个径向输出位置；不作时间或空间插值。定义", "",
             r"\[", r"E_C^{(a,b)}=\max_{(t,r)\in\mathcal S_{ab}}|C_a(t,r)-C_b(t,r)|,\qquad",
             r"E_t^{(a,b)}=|t_{+,a}-t_{+,b}|.", r"\]", "",
             r"正式数值比较门槛为 $E_C\le 2\times10^{-5}\ \mathrm{kg/kg}$、$E_t\le0.1\ \mathrm{s}$；事件夹逼容差复核另按记录中更严的终点差门槛（若为0.02 s则照实列出）判断。温度差记录只作补充，不与含水率门槛混用。", "",
             r"事件定位同时检查 $g(t_-)=M(t_-)-0.15\ge0$、$g(t_+)<0$ 与 $t_+-t_-\le\varepsilon_t$。事件括区宽度衡量固定数值轨道上的定位分辨率，不能代替时间积分误差、空间离散误差或早期初态误差。", "",
             "比较包括0秒和两者共同存在的完整整分输出；终点场各自属于不同的终点时刻，不能把两份终点场之差冒充同一时刻的离散误差。更细算例仍然是数值参考，以上差值不构成未知精确解的严格误差上界，也不是内部实测验证。", "",
             "## 图8：独立改变步长与网格", "",
             f"![图8 第三问终点时间的加密比较]({Path(figure_path).resolve().as_posix()})", "",
             "时间步系列固定同一正式Q2末态；空间系列则让各个N使用各自原网格上的Q2末态，并匹配Q2早期时间尺度。原正式N=960的Q2末态来源单独画为空心菱形，不连接到空间系列中，也不把不同初态来源的变化全归于网格。图中纵轴仅减去一个共同常数以便读数，该常数没有真解含义。", ""]
    if missing:
        lines.extend(["尚未存在的比较记录：" + "、".join(f"`q3_convergence_{n}.json`" for n in missing) + "。", ""])
    lines.extend(["## 已保存算例与初态来源", "",
                  "| 编号 | N | 续算Δt/s | 事件步长/s | 夹逼容差/s | Q2末态来源/早期τ/s | 终点/s | 夹逼宽度/s |",
                  "|---|---:|---:|---:|---:|---|---:|---:|"])
    for run in evidence["ordered_runs"]:
        meta = run["metadata"]
        seed_kind = "原正式Q2" if run.get("source") == "production" else "各网格原生"
        if run.get("base_path"):
            seed_kind += "，局部事件重积分"
        bracket = meta.get("bracket", run.get("checks", {}).get("event_bracket", {}))
        width = bracket.get("width")
        width_text = f"{float(width):.8g}" if width is not None else "未记录"
        tight = "，Picard容差×0.01" if run.get("tight") else ""
        lines.append(f"| {evidence['labels'][run['label']]} | {run['cells']} | {float(run['dt']):g} | "
                     f"{float(run['event_dt']):g} | {float(run['event_tol']):g} | {seed_kind}/{_fraction(_seed_scale(run))}{tight} | "
                     f"{float(run['finish_time_s']):.9f} | {width_text} |")
    if not evidence["ordered_runs"]:
        lines.append("| — | — | — | — | — | 尚无完成算例 | — | — |")
    lines.extend(["", "同一编号只引用其存储的实际终点；没有对终点时间作Richardson外推。表中事件步长采用结果元数据中的实际值：当续算步长小于请求的事件步长时，求解器会相应缩小事件步长，不能把请求值当作实际值标注。", "",
                  "## 实际加密差值", "",
                  "| 检查 | 比较 | maxΔC/(kg/kg) | 最大差位置(t/s,r/cm) | Δ终点/s | 限值C/终点s | 记录结论 |",
                  "|---|---|---:|---|---:|---|---|"])
    for comparison in evidence["comparisons"]:
        lines.append(_comparison_row(comparison, evidence))
    if not evidence["comparisons"]:
        lines.append("| — | — | — | — | — | — | 尚无比较记录 |")
    lines.extend(["", "表中“通过”只对应这一对算例和这一行的门槛；最终选档须综合时间、空间、Q2末态来源、事件积分与Picard检查。各报告顶层的全矩阵结果可能因保留粗档失败而为false，不能据此覆盖细档的逐项结论。", "",
                  "局部事件复核从同一完整锚点重新积分，并复用相同的此前轨迹；其共同整分输出差为0可能正是共享前缀的结果，不能用来证明全程时间离散已收敛。应分别阅读局部事件差值和整条轨迹的时间步比较。", "",
                  "## 水量与末态复核", "",
                  r"使用初始干物质质量口径，独立重组保存字段 $B(t)=\overline C(t)+L_{\mathrm{cum}}(t)-2.55$；水量残差门槛为 $10^{-8}\ \mathrm{kg/kg}$。", "",
                  "| 算例 | 最大水量残差/(kg/kg) | 末态独立最大值/(kg/kg) | 严格低于0.15 | 全场/事件复核 |",
                  "|---|---:|---:|---|---|"])
    for run in evidence["ordered_runs"]:
        check = run.get("checks", {})
        water = check.get("maximum_water_balance_residual")
        maximum = check.get("strict_final_max_C")
        wtext = f"{float(water):.8e}" if water is not None else "未记录"
        mtext = f"{float(maximum):.17g}" if maximum is not None else "未记录"
        strict = "是" if maximum is not None and float(maximum) < .15 else "未确认"
        passed = "通过" if check.get("passed") is True else "未确认"
        lines.append(f"| {evidence['labels'][run['label']]} | {wtext} | {mtext} | {strict} | {passed} |")
    if not evidence["ordered_runs"]:
        lines.append("| — | — | — | — | 尚无末态复核记录 |")
    core = evidence["core"]
    if isinstance(core, dict):
        core = core.get("core", core)
        if core.get("passed") is True:
            lines.extend(["", "已读取核心续算检查：原状态与磁盘重启、相同步序的旧新内核、绝对时间驱动及内部最大值检测检查通过。其含义是实现一致性，不是独立物理正确性证明。"])
    formal = evidence["formal_checks"]
    if isinstance(formal, dict) and isinstance(formal.get("deliverables"), dict):
        artifact = formal["deliverables"]
        if artifact.get("passed") is True:
            lines.extend(["", f"正式结果文件检查记录：逐格核对 {artifact.get('all_excel_C_values_checked', '未记录')} 个Excel浓度值、"
                          f"{artifact.get('table5_rows_checked', '未记录')} 行表5，检查通过；此项证明导出与原数组一致。"])
    lines.extend(["", *_reproduction_text(evidence), *_boundary_text(evidence)])
    failures = [(name, label, detail) for name, report in evidence["reports"].items()
                for label, detail in report.get("failed_runs", {}).items()]
    if failures:
        lines.extend(["", "## 尚未成功的运行", ""])
        lines.extend(f"- {name} / {label}：{detail.get('type', '')}，{detail.get('message', '')}" for name, label, detail in failures)
    lines.extend(["", "## 证据索引", ""])
    for name in evidence["reports"]:
        path = evidence["root"] / "validation" / f"q3_convergence_{name}.json"
        lines.append(f"- [{path.name}]({path.resolve().as_posix()})")
    for name in ("q3_selected_config.json", "q3_acceptance.json", "q3_reproduction.json", "q3_boundary_sensitivity.json"):
        path = evidence["root"] / "validation" / name
        if path.is_file():
            lines.append(f"- [{name}]({path.resolve().as_posix()})")
    prefix_path = evidence["root"] / "paper_appendix_minimal/reproduced/q3_prefix/prefix_comparison.json"
    if prefix_path.is_file():
        lines.append(f"- [独立Q2前段来源与比较记录]({prefix_path.resolve().as_posix()})")
    for run in evidence["ordered_runs"]:
        lines.append(f"- {evidence['labels'][run['label']]}：`{run['label']}`；"
                     f"Q2来源 `{run['source_path']}`；结果 `{run['path']}`。")
        if run.get("base_path"):
            lines.append(f"  局部事件复核复用的全程基线：`{run['base_path']}`。")
    lines.extend(["", "模型仍采用既定环境延拓、固定半径一维径向近似及有效传质口径，不显式计入潜热与端部输运。即使所有规定数值检查通过，也不据此声称这些建模假设已被内部实验数据验证。", ""])
    return "\n".join(lines)


def build_report(root=ROOT, output=None, figures=None, plots=True):
    root = Path(root)
    output = Path(output) if output else root / "validation/问题3精度验证报告.md"
    figures = Path(figures) if figures else root / "q3/figures"
    evidence = load_evidence(root)
    if plots:
        figure8(evidence, figures)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(evidence, figures / "fig8_q3_convergence.png"), encoding="utf-8", newline="\n")
    return {"report": str(output), "figure": str(figures / "fig8_q3_convergence.pdf") if plots else None,
            "matrices_read": list(evidence["reports"]), "completed_runs": len(evidence["runs"]),
            "comparisons_read": len(evidence["comparisons"]), "selected_config_present": evidence["selected"] is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render saved Q3 verification evidence without simulation")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.output, args.figures, not args.no_plots), ensure_ascii=False, indent=2))
