from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse,json
from common.data_loader import load_environment
from common.solver import solve,save_solution
from q1.config import CONFIG as Q1
from q2.config import CONFIG as Q2
p=argparse.ArgumentParser();p.add_argument("--q",type=int,default=1);p.add_argument("--n",type=int,default=40)
p.add_argument("--dt",type=float,default=1.0);p.add_argument("--end",type=int);p.add_argument("--tight",action="store_true")
a=p.parse_args();config=dict(Q1 if a.q==1 else Q2)
if a.end:config["end_time"]=a.end
if a.tight:config["atol"]*=0.01;config["rtol"]*=0.01
result=solve(config,load_environment(bootstrap.ROOT/"data/attachment1.xlsx"),a.n,a.dt)
tag=f"p10cap4_q{a.q}_N{a.n}_dt{a.dt:.12g}"+("_tight" if a.tight else "")+(f"_end{a.end}" if a.end else "")
save_solution(result,bootstrap.ROOT/"validation/runs"/(tag+".npz"))
print(json.dumps(result["metadata"],indent=2),flush=True)
print("final center/surface Celsius",result["temperature_K"][-1,[0,-1]]-273.15,flush=True)
print("final center/surface C",result["moisture"][-1,[0,-1]],flush=True)
