"""Keep runtime writes inside this project; local vendored dependencies are optional."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
for name in ("tmp", "cache/matplotlib", "cache/numba"):
    (RUNTIME / name).mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "cache/matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(RUNTIME / "cache/numba")
os.environ["TEMP"] = os.environ["TMP"] = str(RUNTIME / "tmp")
sys.dont_write_bytecode = True
local_deps = RUNTIME / "deps"
if local_deps.exists():
    sys.path.insert(0, str(local_deps))
