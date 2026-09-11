from pathlib import Path
import json, time
import numpy as np
from common.finite_volume import geometry, integrate_block

def solve(config, environment, cells, dt, sample_count=21, progress=True):
    started=time.perf_counter()
    faces,centers,volumes=geometry(cells,config["radius"])
    radii=np.linspace(0,config["radius"],sample_count)
    end=int(config["end_time"])
    T=np.full(cells,config["initial_T"])
    C=np.full(cells,config["initial_C"])
    outputT=np.empty((end+1,sample_count));outputC=np.empty_like(outputT)
    outputT[0]=config["initial_T"];outputC[0]=config["initial_C"]
    logs=np.zeros((end,9))
    last_report=time.perf_counter()
    for start in range(0,end,60):
        stop=min(start+60,end)
        T,C,ot,oc,log=integrate_block(T,C,float(start),stop,dt,config["question"],
            environment,faces,centers,volumes,radii,config["h"],config["hm"],
            config["atol"],config["rtol"],config["max_iterations"],config.get("time_growth",0.0))
        outputT[start+1:stop+1]=ot;outputC[start+1:stop+1]=oc;logs[start:stop]=log
        if progress and time.perf_counter()-last_report>30:
            print(f'q{config["question"]} N={cells} dt={dt:g}: {stop}/{end} s',flush=True)
            last_report=time.perf_counter()
    cumulative=np.cumsum(logs[:,1])
    balance=logs[:,0]+cumulative-config["initial_C"]
    meta={**config,"mesh":"polynomial10_surface_graded","cells":cells,"dt_scale":dt,"dt_max":min(1.0,4.0*dt,dt*(1+end/config["time_growth"])**2) if config.get("time_growth",0)>0 else dt,"sample_count":sample_count,
          "elapsed_seconds":time.perf_counter()-started,"steps":int(logs[:,2].sum()),
          "iterations":int(logs[:,3].sum()),"rejections":int(logs[:,4].sum()),
          "damped_iterations":int(logs[:,8].sum()),
          "max_scaled_residual":float(logs[:,7].max()),
          "max_water_balance_residual":float(np.max(np.abs(balance)))}
    return {"times":np.arange(end+1,dtype=float),"radii":radii,"temperature_K":outputT,
            "moisture":outputC,"faces":faces,"centers":centers,"volumes":volumes,
            "final_T":T,"final_C":C,"logs":logs,"water_balance":balance,"metadata":meta}

def save_solution(result,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:v for k,v in result.items() if k!="metadata"},
                        metadata_json=json.dumps(result["metadata"],ensure_ascii=False))
    path.with_suffix(".json").write_text(json.dumps(result["metadata"],ensure_ascii=False,indent=2),encoding="utf-8")

def load_solution(path):
    with np.load(path,allow_pickle=False) as data:
        result={k:data[k] for k in data.files if k!="metadata_json"}
        result["metadata"]=json.loads(str(data["metadata_json"]))
        return result
