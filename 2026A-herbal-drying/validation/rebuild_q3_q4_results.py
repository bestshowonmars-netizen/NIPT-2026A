"""Build solution-only Q3/Q4 deliverables from a fresh, checked integration.

This script does not edit a manuscript or rerun Q1/Q2. The --preview option is
explicitly labelled and cannot publish a recomputation claim to the project.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone

import numpy as np
from common.solver import load_solution
from common.drying_result_notes import mean_crossing_interval
from q3.analysis import analyze as analyze3
from q4.analysis import analyze as analyze4
from validation.q3_checks import check_solution as check3, check_deliverables as export_check3
from validation.q4_checks import check_solution as check4, check_deliverables as export_check4
from validation.q3_acceptance import validate_selected as acceptance3
from validation.q4_acceptance import validate_selected as acceptance4
from validation.evidence_paths import project_path

ROOT = bootstrap.ROOT
FRESH = ROOT / 'validation/redone_20260912'
DEFAULT_OUTPUT = ROOT.parents[1] / 'output/第三第四问重新解答_20260912'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compare_arrays(fresh, previous):
    keys = set(fresh) - {'metadata'}
    if keys != set(previous) - {'metadata'}:
        raise AssertionError('Fresh integration changed the saved state fields')
    report = {}
    for key in sorted(keys):
        a, b = np.asarray(fresh[key]), np.asarray(previous[key])
        numeric = a.dtype.kind in 'biufc'
        equal = np.array_equal(a, b, equal_nan=True) if numeric else np.array_equal(a, b)
        if not equal:
            raise AssertionError(f'Fresh array differs from validated reference: {key}; investigate before publishing')
        finite = np.isfinite(a) if numeric else np.zeros(a.shape, dtype=bool)
        delta = float(np.max(abs(a[finite].astype(float) - b[finite].astype(float)))) if np.any(finite) else 0.0
        report[key] = {'shape': list(a.shape), 'equal_including_nan_mask': bool(equal),
                       'maximum_absolute_difference': delta}
    return report


def verify_fresh_receipt(q, path, solution):
    """Bind a completed run receipt to the exact new NPZ, inputs, and settings."""
    launch_path = FRESH / f'q{q}/launch.json'
    receipt_path = FRESH / f'q{q}/receipt.json'
    launch = json.loads(launch_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    meta = solution['metadata']
    if q == 3:
        assert launch['recompute'] is True
        assert receipt['status'] == 'complete'
        assert receipt['fresh_forward_recompute'] is True and receipt['cache_reuse'] is False
        assert receipt['fresh_solution_checks']['passed'] is True
        assert receipt['source_and_kernel_hashes_unchanged'] is True
        assert project_path(launch['output'], ROOT).resolve() == project_path(receipt['output'], ROOT).resolve() == path.resolve()
        assert project_path(launch['source'], ROOT).resolve() == project_path(receipt['source'], ROOT).resolve() == (ROOT / 'q2/solution.npz').resolve()
        for mkey, lkey in (('cells', 'cells'), ('dt_max', 'dt_max_s'), ('event_dt', 'event_dt_s'), ('epsilon_t', 'event_tolerance_s')):
            assert meta[mkey] == launch[lkey] == receipt[lkey]
        expected = receipt['new_solution_sha256']
        started, finished = launch['started_at_utc'], receipt['finished_at_utc']
    else:
        assert launch['status'] == 'computed_pending_checks'
        assert launch['parameters']['recompute'] is True
        assert receipt['passed'] is True and receipt['fresh_solve_from_initial_state'] is True
        assert receipt['cached_solution_used_to_initialize'] is False
        assert receipt['solution_checks']['passed'] is True and receipt['core_checks_passed'] is True
        assert receipt['original_files_unchanged'] is True
        recorded = project_path(launch['output_path'], ROOT)
        assert recorded.resolve() == path.resolve()
        for key in ('cells', 'dt_max', 'dt_scale', 'time_growth', 'event_dt', 'shrinking'):
            assert meta[key] == launch['parameters'][key] == receipt['configuration'][key]
        assert meta['epsilon_t'] == launch['parameters']['event_tol'] == receipt['configuration']['event_tol']
        expected = receipt['fresh_solution_sha256']
        assert launch['fresh_solution_sha256'] == expected
        started, finished = launch['started_utc'], receipt['verified_utc']
    assert meta['completed'] and meta['status'] == 'complete'
    assert meta['finish_time_s'] == receipt['finish_time_s']
    assert digest(path) == expected
    assert datetime.fromisoformat(started) < datetime.fromisoformat(finished)
    journal = receipt.get('interval_journal')
    if journal is not None:
        assert journal['completed'] is True
        assert receipt['durable_driver_sha256'] == digest(ROOT / 'validation/recompute_durable.py')
        checkpoint_check = json.loads((FRESH / 'journal_verification.json').read_text(encoding='utf-8'))
        assert checkpoint_check['passed'] is True and checkpoint_check['questions'][str(q)]['passed'] is True
    return {'passed': True, 'npz_sha256': expected, 'launch_sha256': digest(launch_path),
            'receipt_sha256': digest(receipt_path), 'started_utc': started, 'verified_utc': finished,
            'complete_run_and_parameters_bound_to_npz': True, 'interval_journal': journal}


def table_rows(path):
    text = Path(path).read_text(encoding='utf-8')
    return '\n'.join(line for line in text.splitlines() if line.startswith('|'))


def replace_tokens(template, values):
    for key, value in values.items():
        template = template.replace('{{' + key + '}}', str(value))
    if re.search(r'\{\{[A-Z][A-Z0-9_]*\}\}', template):
        raise AssertionError('Unfilled solution template')
    return template


def solution_values(q, s, other, fixed, out):
    m = mean_crossing_interval(s)
    values = {
        'FINISH_H4': f"{s['metadata']['finish_time_h']:.4f}",
        'CENTER_C': f"{s['moisture'][-1, 0]:.4f}",
        'SURFACE_C': f"{(s['moisture'][-1, -1] if q == 3 else s['surface_moisture'][-1]):.4f}",
        'MEAN_C': f"{s['mean_C'][-1]:.6f}", 'LOSS_C': f"{s['loss_cumulative'][-1]:.6f}",
        'WATER': f"{np.max(abs(s['water_balance'])):.3e}",
        'MEAN_TIME_RANGE': f"{m['t_minus_s']/3600:.4f}—{m['t_plus_s']/3600:.4f}",
        'EARLY_RANGE': f"{m['earlier_min_h']:.4f}—{m['earlier_max_h']:.4f}",
        'EARLY_H2': f"{m['earlier_midpoint_h_for_display']:.2f}",
        'TABLE': table_rows(out / f'q{q}/tables/table{q+2}.md'),
    }
    if q == 4:
        fixed_h = float(fixed['metadata']['finish_time_h'])
        finish_h = float(s['metadata']['finish_time_h'])
        values.update(RADIUS_CM=f"{s['radius_m'][-1]*100:.4f}",
                      Q3_FINISH_H4=f"{other['metadata']['finish_time_h']:.4f}",
                      FIXED_H3=f'{fixed_h:.3f}', REDUCTION_H3=f'{fixed_h-finish_h:.3f}',
                      REDUCTION_PCT1=f'{100*(1-finish_h/fixed_h):.1f}')
    return values


def validation_note(report):
    preview = report['status'] != 'fresh_integrations_verified'
    lines = ['# 第三、第四问重算与验证说明', '',
             f"本轮状态：{report['status']}。生成时间：{report['created_utc']}。", '',
             ('内部预览使用上次正式解；下面的数值不作为本轮重新积分完成的声明。' if preview else
              '第三问从现有第二问10800 s完整状态重新积分；第四问从题设均匀初态重新积分。') +
             '第四问固定半径对照使用已通过来源及加密核验的保存结果。第二问前段和固定对照不标记为本轮重新计算。', '',
             '| 检查 | 第三问 | 第四问 |', '|---|---:|---:|']
    s3, s4 = report['questions']['3'], report['questions']['4']
    lines += [f"| 达标时间/h | {s3['finish_time_h']:.4f} | {s4['finish_time_h']:.4f} |",
              f"| {'参考解' if preview else '本轮'}接受时间步数 | {s3['accepted_steps']} | {s4['accepted_steps']} |",
              '| 原正式解全部数组及NaN位置 | 完全一致 | 完全一致 |',
              '| 全域判据及终点状态重新积分核查 | 通过 | 通过 |',
              '| 每个Excel数值、域外空白、规定表格 | 通过 | 通过 |',
              f"| 最大水量残差/(kg/kg) | {s3['water_max']:.3e} | {s4['water_max']:.3e} |", '',
              '数值内核和输入与原通过验收版本的哈希一致。重算一致性检验的是可复现性，不是物理模型正确性的独立证明。', '',
              '下表重新读取原始加密算例计算差值，未在本轮把整个加密矩阵重复求解。', '',
              '| 问题/算例 | 比较 | 固定条件 | 终点差/s |', '|---|---|---|---:|']
    comparisons3 = report['acceptance']['3']['comparisons_recomputed_from_raw_arrays']
    for c in comparisons3:
        if c['kind'] == 'time':
            lines.append(f"| 第三问 | 时间步1/16→1/32 s | N=960，同一第二问末态 | {c['finish_time_s']['difference']:.6f} |")
        elif c['kind'].startswith('space') and 'N960_' in c['coarse'] and 'N1280_' in c['fine']:
            lines.append(f"| 第三问 | 网格960→1280 | 0.25 s，各网格对应的第二问末态 | {c['finish_time_s']['difference']:.6f} |")
    for c in report['acceptance']['4']['comparisons']:
        if c['kind'] not in ('time', 'space', 'fixed_time', 'fixed_space'):
            continue
        label = '第四问收缩' if c['scope'] == 'shrinking' else '附录4固定对照'
        a, b = c['coarse_config'], c['fine_config']
        kind = c['kind'].removeprefix('fixed_')
        spec = f"时间步{a['dt_max']:g}→{b['dt_max']:g} s" if kind == 'time' else f"网格{a['cells']}→{b['cells']}"
        condition = f"N={a['cells']}" if kind == 'time' else f"时间步{a['dt_max']:g} s"
        lines.append(f"| {label} | {spec} | {condition} | {c['finish_time_s']['difference']:.6f} |")
    lines += ['', '第三、四问主模型所列最终验收比较采用0.1 s终点差门槛；固定半径对照单独采用1 s门槛。'
              '不将较粗算例的失败记录删掉，也不概括为所有加密差均小于0.1 s。', '',
              '实际终点区间及未舍入最大值：', '',
              '| 问题 | 未达标时刻/s | 达标时刻/s | 区间宽度/s | 达标时刻最大含水率/(kg/kg) |',
              '|---|---:|---:|---:|---:|']
    for q in ('3', '4'):
        row = report['questions'][q]
        b = row['bracket']
        lines.append(f"| {q} | {b['t_minus']:.11f} | {b['t_plus']:.11f} | {b['width']:.11f} | {row['max_C']:.17g} |")
    lines += ['', '以上区间由重新积分得到，精细小数用于程序核验；不声称真实干燥时间具有相同精度。', '',
              '平均含水率判据对照仅使用相邻一分钟保存值，不对其做精确事件声明：', '',
              '| 问题 | 平均值跨阈时间区间/h | 相对全域判据提前区间/h |', '|---|---:|---:|']
    for q in ('3', '4'):
        m = report['questions'][q]['mean_crossing']
        lines.append(f"| {q} | {m['t_minus_s']/3600:.6f}—{m['t_plus_s']/3600:.6f} | {m['earlier_min_h']:.6f}—{m['earlier_max_h']:.6f} |")
    lines += ['', '第三问0—3 h未保存完整网格最大值历史，因此配图保留空缺；第四问域外取样点留空，实际表面另列。'
              '温度在两问中始终参与求解。', '',
              '两问沿用有效水分驱动、一维径向近似及不显式计入潜热的假设；潜热影响尚未定量评估。'
              '当前数据没有内部温湿场实测值，不能用水量残差或加密差代替实验验证。', '',
              '完整机器核查：[validation/recomputed.json](validation/recomputed.json)。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='重新生成第三、四问解答、图表和验证包，不制作论文')
    parser.add_argument('--q3', type=Path, default=FRESH / 'q3/solution.npz')
    parser.add_argument('--q4', type=Path, default=FRESH / 'q4/solution.npz')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--preview', action='store_true', help='只以旧解预览，不声明重算、不发布到工程')
    parser.add_argument('--publish-project', action='store_true')
    args = parser.parse_args()
    if args.preview and args.publish_project:
        raise ValueError('Preview cannot be published')
    qpaths = {3: args.q3, 4: args.q4}
    if args.preview:
        qpaths = {q: ROOT / f'q{q}/solution.npz' for q in (3, 4)}
        out = FRESH / 'preview_delivery'
    else:
        out = args.output.resolve()
        for q, path in qpaths.items():
            if path.resolve() == (ROOT / f'q{q}/solution.npz').resolve():
                raise ValueError('A fresh integration path is required; use --preview for cached values')
            launch = FRESH / f'q{q}/launch.json'
            receipt = FRESH / f'q{q}/receipt.json'
            if not launch.exists() or not receipt.exists():
                raise FileNotFoundError(f'Await completed independent integration receipt: {receipt}')
    out.mkdir(parents=True, exist_ok=True)
    solutions = {q: load_solution(path) for q, path in qpaths.items()}
    previous = {q: load_solution(ROOT / f'q{q}/solution.npz') for q in (3, 4)}
    fixed = load_solution(ROOT / 'q4/fixed_control.npz')
    report = {'status': 'preview_using_previous_arrays' if args.preview else 'fresh_integrations_verified',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'questions': {}, 'acceptance': {}}
    for q in (3, 4):
        s = solutions[q]
        print(f'Checking Q{q} recomputed arrays and event states', flush=True)
        checks = check3(s, load_solution(ROOT / 'q2/solution.npz'), require_dense=True) if q == 3 else check4(s)
        report['questions'][str(q)] = {
            'source': str(qpaths[q].resolve()), 'source_sha256': digest(qpaths[q]),
            'baseline_sha256': digest(ROOT / f'q{q}/solution.npz'),
            'all_array_comparison': compare_arrays(s, previous[q]), 'solution_checks': checks,
            'finish_time_h': float(s['metadata']['finish_time_h']),
            'accepted_steps': int(s['metadata']['steps']), 'max_C': float(s['max_C'][-1]),
            'water_max': float(np.max(abs(s['water_balance']))),
            'bracket': s['metadata']['bracket'], 'mean_crossing': mean_crossing_interval(s),
        }
        if not args.preview:
            report['questions'][str(q)]['fresh_receipt_binding'] = verify_fresh_receipt(q, qpaths[q], s)
    print('Rechecking raw refinement evidence and fixed-radius control', flush=True)
    report['acceptance']['3'] = acceptance3(solutions[3])
    report['acceptance']['4'] = acceptance4()
    assert report['acceptance']['3']['passed'] and report['acceptance']['4']['passed']
    comparisons = {'q3': solutions[3], 'a4_fixed': fixed}
    analyze3(solutions[3], out / 'q3', result_path=out / 'result3.xlsx', plots=True)
    analyze4(solutions[4], out / 'q4', result_path=out / 'result4.xlsx', plots=True,
             comparison_solutions=comparisons)
    for q in (3, 4):
        check = export_check3 if q == 3 else export_check4
        exported = check(solutions[q], out / f'result{q}.xlsx', out / f'q{q}/tables/table{q+2}.csv',
                         out / f'q{q}/endpoint.json')
        report['questions'][str(q)]['export_checks'] = exported
        template = (ROOT / f'validation/solution_templates/q{q}.md').read_text(encoding='utf-8')
        text = replace_tokens(template, solution_values(q, solutions[q], solutions[3], fixed, out))
        if args.preview:
            text = '> 内部预览：使用上次已验证数据，本轮重算尚未完成。\n\n' + text
        (out / f'第{q}问解答.md').write_text(text, encoding='utf-8')
        shutil.copy2(qpaths[q], out / f'q{q}/solution.npz')
        if not args.preview:
            shutil.copy2(FRESH / f'q{q}/launch.json', out / f'q{q}/launch.json')
            shutil.copy2(FRESH / f'q{q}/receipt.json', out / f'q{q}/receipt.json')
            if q == 4:
                shutil.copy2(FRESH / 'q4/preflight.json', out / 'q4/preflight.json')
    shutil.copy2(ROOT / 'q4/fixed_control.npz', out / 'q4/fixed_control.npz')
    report['fixed_control'] = {'recomputed_this_turn': False, 'source_sha256': digest(ROOT / 'q4/fixed_control.npz'),
                               'finish_time_h': float(fixed['metadata']['finish_time_h'])}
    write_json(out / 'validation/recomputed.json', report)
    if not args.preview and (FRESH / 'journal_verification.json').exists():
        shutil.copy2(FRESH / 'journal_verification.json', out / 'validation/journal_verification.json')
    (out / '验证说明.md').write_text(validation_note(report), encoding='utf-8')
    (out / 'README.md').write_text(
        '# 第三、第四问解答与结果\n\n'
        '先阅读[第三问解答](第3问解答.md)和[第四问解答](第4问解答.md)。'
        '结果文件为result3.xlsx、result4.xlsx；q3、q4目录包含未舍入状态、规定表格、精确终点和配图。\n\n'
        '配图提供PNG、PDF、SVG版本；PDF是单张计算结果图，不是论文。'
        '方法短稿在各问目录的“方法草稿与结果解释.md”。\n\n'
        '本轮积分与核查见[验证说明](验证说明.md)。完整实现保留在NIPT-2026A工程；'
        '生成入口为validation/rebuild_q3_q4_results.py，实际重算入口和参数见q3、q4的launch.json。\n', encoding='utf-8')
    if args.publish_project:
        for q in (3, 4):
            shutil.copy2(out / f'result{q}.xlsx', ROOT / f'submission_results/result{q}.xlsx')
            for part in ('figures', 'tables'):
                for src in (out / f'q{q}/{part}').iterdir():
                    if src.is_file():
                        shutil.copy2(src, ROOT / f'q{q}/{part}' / src.name)
            shutil.copy2(out / f'q{q}/方法草稿与结果解释.md', ROOT / f'q{q}/方法草稿与结果解释.md')
            text = (out / f'第{q}问解答.md').read_text(encoding='utf-8')
            text = text.replace(f'(result{q}.xlsx)', f'(submission_results/result{q}.xlsx)')
            text = text.replace('(验证说明.md)', '(validation/redone_20260912/验证说明.md)')
            (ROOT / f'第{q}问解答.md').write_text(text, encoding='utf-8')
        (FRESH / '验证说明.md').write_text(validation_note(report).replace('(validation/recomputed.json)', '(recomputed.json)'), encoding='utf-8')
        write_json(FRESH / 'recomputed.json', report)
    manifest = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', manifest)
    if not args.preview:
        archive = out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
        print(f'Delivery complete: {out}\nZIP: {archive}', flush=True)
    else:
        print(f'Internal preview only: {out}', flush=True)


if __name__ == '__main__':
    main()
