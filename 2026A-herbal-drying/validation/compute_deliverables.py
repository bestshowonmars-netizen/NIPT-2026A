from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap,concurrent.futures

def worker(q):
    from run_all import run_question
    return run_question(q,recompute=True,plots=False,export=False)['metadata']

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(worker,[1,2]):print(result,flush=True)
