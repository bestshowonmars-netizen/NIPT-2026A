from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap,subprocess,concurrent.futures,os

def worker(q):
    root=bootstrap.ROOT
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run([sys.executable,'-B','-X','utf8',str(root/'paper_appendix_minimal/minimal_solver.py'),
        '--question',str(q),'--compute-only'],cwd=root/'paper_appendix_minimal',env=env,check=True)
    return q
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for q in pool.map(worker,[1,2]):print('Appendix independent computation complete',q,flush=True)
