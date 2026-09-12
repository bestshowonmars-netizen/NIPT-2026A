"""Run Q3/Q4 with an exact, atomic journal of newly integrated minute blocks.

The verified numerical sources are not edited. Replaying a block requires an
identical incoming full state, time interval, parameters, and numerical source.
This journal contains this recomputation's work, never a published final state.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import importlib
import json
import os
import runpy
from datetime import datetime, timezone
import numpy as np

ROOT = bootstrap.ROOT
RUN = ROOT / 'validation/redone_20260912'


def signature(args, source_digest):
    h = hashlib.sha256(source_digest.encode('ascii'))
    for arg in args:
        a = np.asarray(arg)
        h.update(str(a.dtype).encode('ascii'))
        h.update(str(a.shape).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()


def install_journal(q):
    module = importlib.import_module(f'q{q}.solve_q{q}')
    original = module.advance_interval
    cache = ROOT / f'.runtime/redone_20260912/q{q}_blocks'
    cache.mkdir(parents=True, exist_ok=True)
    # The input signature also binds environment, geometry and every argument.
    sources = [ROOT / 'common/finite_volume.py', ROOT / 'common/continuation.py',
               ROOT / f'q{q}/solve_q{q}.py', ROOT / f'q{q}/end_time_detector.py']
    if q == 4:
        sources += [ROOT / 'q4/kernel.py', ROOT / 'q4/shrinking_radius.py']
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    stats = {'question': q, 'computed_blocks': 0, 'replayed_blocks': 0,
             'source_digest': source_hash, 'journal_path': str(cache),
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'method': 'Exact input-state keyed atomic minute-block journal; no interpolation'}

    def advance(*args):
        key = signature(args, source_hash)
        path = cache / (key + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert saved['input_signature'].item() == key
                result = (saved['T'].copy(), saved['C'].copy(), saved['diag'].copy())
            stats['replayed_blocks'] += 1
        else:
            result = original(*args)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, input_signature=np.asarray(key),
                                T=result[0], C=result[1], diag=result[2])
            os.replace(temporary, path)
            stats['computed_blocks'] += 1
        stats.update(last_start_s=float(args[2]), last_end_s=float(args[3]),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        target = RUN / f'q{q}/journal_status.json'
        temporary = target.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(stats, indent=2), encoding='utf-8')
        os.replace(temporary, target)
        return result

    module.advance_interval = advance
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', choices=['3', '4'], required=True)
    args = parser.parse_args()
    q = int(args.question)
    stats = install_journal(q)
    folder = RUN / f'q{q}'
    if q == 3:
        runpy.run_path(str(folder / 'recompute.py'), run_name='__main__')
        (folder / 'launch.json').write_bytes((folder / 'started.json').read_bytes())
    else:
        runpy.run_path(str(folder / 'run.py'), run_name='__main__')
        verifier = runpy.run_path(str(folder / 'verify.py'), run_name='q4_verification')
        verifier['verify']()
    receipt_path = folder / 'receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    stats.update(completed=True, completed_utc=datetime.now(timezone.utc).isoformat())
    receipt['interval_journal'] = stats
    receipt['durable_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Q{q} integration and verification complete.', flush=True)


if __name__ == '__main__':
    main()
