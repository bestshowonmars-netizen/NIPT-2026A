"""Check replay of actual newly integrated blocks against the original kernel."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import importlib
import json
import numpy as np
from common.data_loader import load_environment
from common.solver import load_solution
from common.finite_volume import geometry
from q4.shrinking_radius import load_radius_data
from validation.recompute_durable import install_journal

ROOT = bootstrap.ROOT
results = {}
environment = load_environment(ROOT / 'data/attachment1.xlsx')
for q in (3, 4):
    module = importlib.import_module(f'q{q}.solve_q{q}')
    original = module.advance_interval
    meta = load_solution(ROOT / f'q{q}/solution.npz')['metadata']
    if q == 3:
        seed = load_solution(ROOT / 'q2/solution.npz')
        T, C = seed['final_T'].copy(), seed['final_C'].copy()
        faces, centers, volumes = seed['faces'], seed['centers'], seed['volumes']
        start = 10800.0
    else:
        faces, centers, volumes = geometry(meta['cells'], meta['radius'])
        T = np.full(meta['cells'], meta['initial_T'])
        C = np.full(meta['cells'], meta['initial_C'])
        start = 0.0
    args = (T, C, start, start+60.0, meta['dt_max'], environment, faces, centers, volumes,
            meta['h'], meta['hm'], meta['atol'], meta['rtol'], meta['max_iterations'])
    if q == 4:
        radius, slopes = load_radius_data(ROOT / 'data/attachment2.xlsx')
        args += (radius, slopes, True, meta['dt_scale'], meta['time_growth'])
    expected = original(*args)
    stats = install_journal(q)
    actual = module.advance_interval(*args)
    assert stats['computed_blocks'] == 0 and stats['replayed_blocks'] == 1
    errors = {}
    for key, a, b in zip(('T', 'C', 'diagnostics'), expected, actual):
        assert np.array_equal(a, b)
        errors[key] = float(np.max(abs(a-b)))
    results[str(q)] = {'passed': True, 'start_s': start, 'stop_s': start+60,
                        'actual_block_replayed': True, 'fresh_original_kernel_comparison': errors}
path = ROOT / 'validation/redone_20260912/journal_verification.json'
path.write_text(json.dumps({'passed': True, 'questions': results}, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
