from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap,json,concurrent.futures
from validation.grid_convergence import run,difference

def worker(q):
    return run(q,1600,1/1024,growth=600)['metadata']

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for meta in pool.map(worker,[1,2]):print(json.dumps(meta),flush=True)
