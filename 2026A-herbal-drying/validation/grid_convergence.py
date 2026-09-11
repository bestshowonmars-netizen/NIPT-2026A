"""Independent space/time refinement on all integer seconds and 21 required radii."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse,json,time
import numpy as np
from common.solver import solve,save_solution,load_solution
from common.data_loader import load_environment
from q1.config import CONFIG as Q1
from q2.config import CONFIG as Q2
ROOT=bootstrap.ROOT

def run(q,n,dt,tight=False,growth=0.0,force=False):
    tag=f"p10cap4_q{q}_N{n}_dt{dt:.12g}"+(f"_growth{growth:g}" if growth else "")+("_tight" if tight else "")
    path=ROOT/"validation/runs"/(tag+".npz")
    if path.exists() and not force:return load_solution(path)
    cfg=dict(Q1 if q==1 else Q2)
    cfg["time_growth"]=growth
    if tight:cfg["atol"]*=.01;cfg["rtol"]*=.01
    print(f"START {tag}",flush=True)
    data=solve(cfg,load_environment(ROOT/"data/attachment1.xlsx"),n,dt)
    save_solution(data,path)
    print(f"DONE {tag} {data['metadata']['elapsed_seconds']:.2f}s balance={data['metadata']['max_water_balance_residual']:.3g}",flush=True)
    return data

def difference(coarse,fine):
    result={}
    for key,label in [("temperature_K","T"),("moisture","C")]:
        delta=np.abs(coarse[key]-fine[key])
        ij=np.unravel_index(np.argmax(delta),delta.shape)
        result[label]={"maximum":float(delta[ij]),"time_s":int(ij[0]),"radius_cm":float(fine["radii"][ij[1]]*100)}
    return result

def measure(q,kind,n0,dt0,n1,dt1):
    result={"q":q,"kind":kind,"coarse":{"N":n0,"dt":dt0},"fine":{"N":n1,"dt":dt1},
            **difference(run(q,n0,dt0),run(q,n1,dt1))}
    print(json.dumps(result),flush=True)
    return result

def initial(q):
    records=[]
    for a,b in [(40,80),(80,160)]:records.append(measure(q,"space",a,.25,b,.25))
    for a,b in [(1.,.5),(.5,.25)]:records.append(measure(q,"time",160,a,160,b))
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--q",type=int,choices=[1,2]);p.add_argument("--initial",action="store_true")
    args=p.parse_args();records=[]
    for q in ([args.q] if args.q else [1,2]):records.extend(initial(q))
    (ROOT/"validation/p10_initial_convergence.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
