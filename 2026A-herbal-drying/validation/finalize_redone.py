"""Finish this running computation's checked exports once both receipts exist."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import json
import subprocess
import time
from datetime import datetime, timezone

ROOT = bootstrap.ROOT
RUN = ROOT / 'validation/redone_20260912'
deadline = time.monotonic() + 7200
while time.monotonic() < deadline:
    ready = True
    for q in (3,4):
        p = RUN / f'q{q}/receipt.json'
        try:
            receipt = json.loads(p.read_text(encoding='utf-8'))
            ready = ready and receipt.get('interval_journal', {}).get('completed', False)
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
    if ready:
        break
    time.sleep(5)
else:
    raise TimeoutError('Both fresh integration receipts are not yet complete; no final delivery claimed')
with (RUN / 'delivery.log').open('w', encoding='utf-8') as log:
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
                             'validation/rebuild_q3_q4_results.py', '--publish-project'],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
status = {'status': 'complete' if result.returncode == 0 else 'failed',
          'exit_code': result.returncode, 'finished_utc': datetime.now(timezone.utc).isoformat()}
(RUN / 'delivery_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
raise SystemExit(result.returncode)
