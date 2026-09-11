"""Emit a standalone same-algorithm Q4 reproduction, with no project imports."""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]


def build(config=None):
    config = config or json.loads((ROOT/'validation/q4_selected_config.json').read_text(encoding='utf-8'))
    selected = {key: (int(config[key]) if key=='cells' else float(config[key])) for key in
                ('cells','dt_max','dt_scale','event_dt','event_tol')}
    functions = []
    for name in ('shrinking_radius.py', 'kernel.py', 'end_time_detector.py'):
        tree = ast.parse((ROOT/'q4'/name).read_text(encoding='utf-8'))
        functions.extend(ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef))
    header = '''"""Standalone Q4 material-domain solver; see the appendix README."""
from pathlib import Path
import argparse, json, time, hashlib
import minimal_solver as base
import numpy as np
from numba import njit
from openpyxl import load_workbook
from minimal_solver import geometry, links, linear_step, residual, reconstruct, environment_at, load_environment
'''
    driver = r'''

def main(argv=None):
    folder=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description="Standalone appendix-4 reproduction from t=0")
    p.add_argument('--attachment',type=Path,default=folder.parent/'data/attachment1.xlsx')
    p.add_argument('--radius',type=Path,default=folder.parent/'data/attachment2.xlsx')
    p.add_argument('--template',type=Path,default=folder.parent/'data/templates/result4_template.xlsx')
    p.add_argument('--output',type=Path,default=folder/'reproduced/q4')
    p.add_argument('--cells',type=int,default=SELECTED['cells'])
    p.add_argument('--dt-max',type=float,default=SELECTED['dt_max'])
    p.add_argument('--dt-scale',type=float,default=SELECTED['dt_scale'])
    p.add_argument('--event-dt',type=float,default=SELECTED['event_dt'])
    p.add_argument('--event-tol',type=float,default=SELECTED['event_tol'])
    p.add_argument('--fixed-radius',action='store_true')
    args=p.parse_args(argv)
    if args.cells<2 or min(args.dt_max,args.dt_scale,args.event_dt,args.event_tol)<=0:
        raise ValueError('Positive steps and at least two cells are required')
    args.output.mkdir(parents=True,exist_ok=True)
    env=load_environment(args.attachment); rd,rs=load_radius_data(args.radius)
    faces,centers,V=geometry(args.cells,.02)
    T,C=np.full(args.cells,301.15),np.full(args.cells,2.55)
    radii=np.linspace(0,.02,21)
    cfg=dict(h=25.,hm=8e-7,threshold=.15,shrinking=not args.fixed_radius,
             atol=1e-10,rtol=1e-11,max_iterations=60,dt_scale=args.dt_scale,
             time_growth=600.,event_dt=min(args.event_dt,args.dt_max),epsilon_t=args.event_tol)
    times=[];Ts=[];Cs=[];surfaceT=[];surfaceC=[];rhistory=[];masks=[]
    maxima=[];locations=[];balances=[];losses=[]
    def record(t,loss):
        st,sc,mask=sample_fields(T,C,t,env,faces,centers,radii,25.,8e-7,rd,rs,cfg['shrinking'])
        ut,uc=surface_fields(T,C,t,env,faces,centers,25.,8e-7,rd,rs,cfg['shrinking'])
        m,r,_,_=global_maximum(T,C,t,env,faces,centers,25.,8e-7,rd,rs,cfg['shrinking'])
        times.append(t);Ts.append(st);Cs.append(sc);masks.append(mask)
        surfaceT.append(ut);surfaceC.append(uc);rhistory.append(radius_at(t,rd,rs,cfg['shrinking']))
        maxima.append(m);locations.append(r);losses.append(loss)
        balances.append(np.dot(V,C)/V.sum()+loss-2.55)
    t,loss=0.,0.;record(t,loss)
    last_print=time.perf_counter();started=last_print;end_limit=72*3600.
    trace=[]
    while t<30*86400.:
        if t>=end_limit:
            end_limit=min(2*end_limit,30*86400.)
            print(f'Extending search to {end_limit/3600:g} h',flush=True)
        stop=t+60.
        tn,cn,diag=advance_interval(T,C,t,stop,args.dt_max,env,faces,centers,V,
            25.,8e-7,1e-10,1e-11,60,rd,rs,cfg['shrinking'],args.dt_scale,600.)
        m,_,_,_=global_maximum(tn,cn,stop,env,faces,centers,25.,8e-7,rd,rs,cfg['shrinking'])
        terminal=False
        if m<.15:
            event=refine_endpoint(T,C,t,stop,env,faces,centers,V,cfg,rd,rs)
            trace.extend(event['trace'].tolist())
            tn,cn,diag,stop=event['T'],event['C'],event['diag'],event['time']
            terminal=event['crossed']
        T,C,t=tn,cn,stop;loss+=diag[1];record(t,loss)
        if terminal:break
        if time.perf_counter()-last_print>30:
            print(f'Q4 appendix: t={t/3600:.3f} h; max C={maxima[-1]:.8f}',flush=True)
            last_print=time.perf_counter()
    else:
        raise RuntimeError('No threshold crossing within computational search bound')
    bracket=event['bracket']
    assert bracket['g_minus']>=0 and bracket['g_plus']<0
    meta=dict(finish_time_s=t,finish_time_h=t/3600,bracket=bracket,cells=args.cells,
        dt_max=args.dt_max,dt_scale=args.dt_scale,event_dt=cfg['event_dt'],event_tol=args.event_tol,
        shrinking=cfg['shrinking'],from_initial=True,standalone_same_algorithm=True,
        maximum_water_balance=float(np.max(np.abs(balances))),runtime_s=time.perf_counter()-started,
        attachment_sha256=hashlib.sha256(args.attachment.read_bytes()).hexdigest(),
        radius_sha256=hashlib.sha256(args.radius.read_bytes()).hexdigest())
    np.savez_compressed(args.output/'q4_independent.npz',times=np.asarray(times),radii=radii,
        temperature_K=np.asarray(Ts),moisture=np.asarray(Cs),domain_mask=np.asarray(masks),
        surface_temperature_K=np.asarray(surfaceT),surface_moisture=np.asarray(surfaceC),
        radius_m=np.asarray(rhistory),max_C=np.asarray(maxima),max_r=np.asarray(locations),
        water_balance=np.asarray(balances),loss_cumulative=np.asarray(losses),
        final_T=T,final_C=C,faces=faces,centers=centers,volumes=V,
        radius_data=rd,radius_slopes=rs,event_trace=np.asarray(trace),metadata_json=json.dumps(meta))
    wb=load_workbook(args.template);ws=wb.active
    for row in ws:
        for cell in row:
            if cell.coordinate!='A1':cell.value=None
    for j in range(21):ws.cell(1,j+2,round(j*.1,1))
    ws.cell(1,23,'药材表面');row=2
    for index,ti in enumerate(times):
        if ti<=0 or abs(ti/60-round(ti/60))>1e-9:continue
        ws.cell(row,1,int(round(ti)))
        for j,value in enumerate(list(Cs[index])+[surfaceC[index]]):
            cell=ws.cell(row,j+2)
            if np.isfinite(value):cell.value=round(float(value),4)
            cell.number_format='0.0000'
        row+=1
    ws.freeze_panes='B2';wb.properties.creator='';wb.properties.lastModifiedBy=''
    wb.save(args.output/'result4.xlsx')
    (args.output/'endpoint.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    return meta

if __name__=='__main__':main()
'''
    target=ROOT/'paper_appendix_minimal/minimal_q4.py'
    target.write_text(header+'\nSELECTED = '+repr(selected)+'\n\n'+'\n\n'.join(functions)+driver,encoding='utf-8')
    print(target)
    return target


if __name__=='__main__':build()
