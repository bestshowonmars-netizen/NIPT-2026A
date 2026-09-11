from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import json,concurrent.futures
from validation.grid_convergence import run,difference

def worker(task):
    q,n,dt,tight,force=task
    data=run(q,n,dt,tight=tight,growth=600,force=force)
    return dict(q=q,N=n,dt=dt,tight=tight,**data['metadata'])

def main(recompute=False):
    tasks=[]
    for q in [1,2]:
        tasks += [(q,n,1/1024,False,recompute) for n in [640,960,1280,1600]]
        tasks += [(q,1280,dt,False,recompute) for dt in [1/512,1/768]]
        tasks += [(q,960,1/768,False,recompute),(q,960,1/768,True,recompute)]
    summaries=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for summary in executor.map(worker,tasks):
            summaries.append(summary)
            (bootstrap.ROOT/'validation/strict_run_summaries.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    records=[]
    for q in [1,2]:
        for a,b in [(640,960),(960,1280),(1280,1600)]:
            d=difference(run(q,a,1/1024,growth=600),run(q,b,1/1024,growth=600))
            records.append(dict(q=q,kind='space',required=(a>=960),coarse=dict(N=a,dt=1/1024),fine=dict(N=b,dt=1/1024),**d))
        for a,b in [(1/512,1/768),(1/768,1/1024)]:
            d=difference(run(q,1280,a,growth=600),run(q,1280,b,growth=600))
            records.append(dict(q=q,kind='time',coarse=dict(N=1280,dt=a),fine=dict(N=1280,dt=b),**d))
        d=difference(run(q,960,1/768,growth=600),run(q,1600,1/1024,growth=600))
        records.append(dict(q=q,kind='joint',coarse=dict(N=960,dt=1/768),fine=dict(N=1600,dt=1/1024),**d))
        d=difference(run(q,960,1/768,growth=600),run(q,960,1/768,tight=True,growth=600))
        records.append(dict(q=q,kind='picard',coarse=dict(N=960,dt=1/768),fine=dict(N=960,dt=1/768),**d))
    for item in records:
        limit=1e-5 if item['kind'] in ['space','time'] else (2e-5 if item['kind']=='joint' else 1e-6)
        item['required']=item.get('required',True)
        item['limit']=limit
        item['passed']=all(item[k]['maximum']<limit for k in ['T','C'])
        print(json.dumps(item),flush=True)
    (bootstrap.ROOT/'validation/strict_convergence.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    if not all(item['passed'] for item in records if item['required']):
        raise RuntimeError('Strict numerical refinement failed; see strict_convergence.json')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--recompute',action='store_true')
    main(recompute=parser.parse_args().recompute)
