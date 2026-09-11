from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import json
from validation.grid_convergence import run,difference
records=[]
for q in [1,2]:
    for n in [320,640,1280]:
        run(q,n,1/128,growth=600)
    for n in [640,1280]:
        d=difference(run(q,n//2,1/128,growth=600),run(q,n,1/128,growth=600))
        print('SPACE',q,n,json.dumps(d),flush=True)
        records.append(dict(q=q,kind='space',N=n,dt_scale=1/128,growth=600,**d))
    for dt in [1/256,1/512]:
        d=difference(run(q,1280,dt*2,growth=600),run(q,1280,dt,growth=600))
        print('TIME',q,dt,json.dumps(d),flush=True)
        records.append(dict(q=q,kind='time',N=1280,dt_scale=dt,growth=600,**d))
(bootstrap.ROOT/'validation/refinement_stage1.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
