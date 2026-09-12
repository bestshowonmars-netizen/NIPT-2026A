"""Concise Q3/Q4 explanations derived from unrounded results, not fitted times."""
from pathlib import Path
import numpy as np


def mean_crossing_interval(solution):
    """Return adjacent saved states around the first strict mean-C crossing."""
    times = np.asarray(solution['times'], dtype=float)
    values = np.asarray(solution['mean_C'], dtype=float)
    threshold = float(solution['metadata']['threshold'])
    if values.shape != times.shape or not np.isfinite(values).all():
        raise ValueError('Mean history must match the finite saved time series')
    if not np.all(np.diff(times) > 0):
        raise ValueError('Saved times must increase')
    indices = np.flatnonzero(values < threshold)
    if not len(indices) or indices[0] == 0:
        raise ValueError('A saved non-dry/dry pair is required')
    hi = int(indices[0])
    lo = hi - 1
    assert values[lo] >= threshold and values[hi] < threshold
    finish = float(solution['metadata']['finish_time_s'])
    return {
        't_minus_s': float(times[lo]), 't_plus_s': float(times[hi]),
        'mean_C_minus': float(values[lo]), 'mean_C_plus': float(values[hi]),
        'width_s': float(times[hi] - times[lo]),
        'earlier_min_h': float((finish - times[hi]) / 3600),
        'earlier_max_h': float((finish - times[lo]) / 3600),
        'earlier_midpoint_h_for_display': float((finish - (times[lo] + times[hi]) / 2) / 3600),
        'method': 'Adjacent stored mean values; no interpolation or event reintegration',
        'criterion_role': 'Illustrates early stopping under a criterion that does not meet the problem requirement',
    }


def write_short_note(solution, folder, comparison_solutions=None):
    """Write a three-paragraph method/result note without a paper template."""
    q = int(solution['metadata']['question'])
    finish = float(solution['metadata']['finish_time_h'])
    center = float(solution['moisture'][-1, 0])
    surface = float(solution['moisture'][-1, -1] if q == 3 else solution['surface_moisture'][-1])
    average = float(solution['mean_C'][-1])
    loss = float(solution['loss_cumulative'][-1])
    crossing = mean_crossing_interval(solution)
    if q == 3:
        text = rf'''# 问题3：方法草稿与结果解释

沿用第二问固定半径的一维径向模型及附录3物性，从3小时的完整、未舍入网格状态续算，保留绝对时间。采用有限体积、全隐式推进与Picard迭代，持续联合求解温度和含水率。检查全部网格单元及中心、表面重构值，以全域最大含水率 \(M(t)<0.15\) 为条件；从同一未达标状态重新积分候选时刻，缩小未达标与达标时间区间，取已达标时刻报告。

达标总时间为{finish:.4f}小时，最后达标位置为中心。终点中心和表面干基含水率分别显示为{center:.4f}、{surface:.4f} kg/kg，平均含水率为{average:.6f} kg/kg，单位初始干物质质量累计失水为{loss:.6f} kg/kg。中心未舍入值已严格小于0.15；0.1500仅是显示舍入。按一分钟保存结果，误用平均含水率判断将提前约{crossing['earlier_midpoint_h_for_display']:.2f}小时停机，说明平均达标不能代替各处达标。

4小时后环境按50℃、0.0500 kg/kg延续，水分指标作药材侧等效驱动；模型不显式计入潜热，其影响尚未定量评估。终点区间宽度不超过0.01 s，仅表示数值定位分辨率；网格与步长误差另行核查。result3.xlsx保持60 s采样，精确非整分钟终点另存记录和规定表格末行。上述结果为有效模型预测。
'''
    elif q == 4:
        control = (comparison_solutions or {}).get('a4_fixed')
        contrast = '收缩作用用附录4固定初始半径对照判断，第三、四问还存在物性差异，不能将两问时长之差全部归因于收缩。'
        if control is not None:
            fixed = float(control['metadata']['finish_time_h'])
            contrast = f'同附录4固定初始半径对照需{fixed:.3f}小时，收缩使本模型预测时长缩短约{(1-finish/fixed)*100:.1f}%。第三、四问还存在物性差异，不能将两问时长之差全部归因于收缩。'
        text = rf'''# 问题4：方法草稿与结果解释

从题设均匀初态独立计算，采用附录4物性和附件2半径的保形插值；假设长度不变、材料径向同比例收缩、各壳干物质质量不变。令 \(s=R(t)/R_0\)、\(x=r/s\)，在固定材料网格上将内部导热与扩散系数除以 \(s^2\)，表面交换系数除以 \(s\)。采用有限体积、全隐式推进及Picard迭代联合求解两场，以全域最大含水率严格低于0.15定位终点。

达标总时间为{finish:.4f}小时，最后达标位置为中心，终点半径为{float(solution['radius_m'][-1])*100:.4f} cm。中心和实际表面含水率分别显示为{center:.4f}、{surface:.4f} kg/kg，平均含水率为{average:.6f} kg/kg，单位初始干物质质量累计失水为{loss:.6f} kg/kg。中心未舍入值严格低于阈值。固定采样位置若超出未舍入的实际半径则留空，表面单独取值。

{contrast}经验密度仅用于有效热容量，水量按不变的材料壳干质量核算；不以另一套密度关系作守恒旁证。环境延续与等效水分驱动沿用既定假设；不显式计入潜热，其影响尚未定量评估。精度由第四问独立加密检查支持，结果不属于实测验证。
'''
    else:
        raise ValueError('Only questions 3 and 4 have these drying notes')
    path = Path(folder) / '方法草稿与结果解释.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path
