from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap,json
from common.plotting import convergence_figure
if __name__=="__main__":
    records=json.loads((bootstrap.ROOT/"validation/strict_convergence.json").read_text(encoding="utf-8"))
    convergence_figure(records,bootstrap.ROOT/"validation/figures")
