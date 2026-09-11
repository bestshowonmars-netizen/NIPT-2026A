"""Run only questions 1/2. All generated paths are anchored to this project."""
import bootstrap
from pathlib import Path
import argparse,json,time
import numpy as np
from common.data_loader import load_environment
from common.solver import solve,save_solution,load_solution
from common.excel_export import export_result
from common.analysis_outputs import write_tables_and_note
from common.plotting import figure1,figure2,figure3,figure4,convergence_figure
from q1.config import CONFIG as Q1
from q2.config import CONFIG as Q2
ROOT=bootstrap.ROOT

def final_config(q):
    cfg=dict(Q1 if q==1 else Q2)
    cfg['time_growth']=600.0
    return cfg,960,1/768

def run_question(q,*,recompute=False,plots=True,export=True,cells=None,dt=None):
    cfg,n,step=final_config(q)
    if cells is not None:n=cells
    if dt is not None:step=dt
    env=load_environment(ROOT/'data/attachment1.xlsx')
    path=ROOT/f'q{q}/solution.npz'
    result=None
    if path.exists() and not recompute:
        candidate=load_solution(path)
        if (all(candidate['metadata'].get(k)==v for k,v in cfg.items())
            and candidate['metadata']['cells']==n and candidate['metadata']['dt_scale']==step
            and candidate['metadata']['sample_count']==101):
            result=candidate
    if result is None:
        print(f'Compute question {q}: N={n}, dt_scale={step:.12g} s',flush=True)
        result=solve(cfg,env,n,step,sample_count=101)
        save_solution(result,path)
    if export:
        export_result(result,ROOT/f'data/templates/result{q}_template.xlsx',ROOT/f'submission_results/result{q}.xlsx')
        write_tables_and_note(result,q,ROOT/f'q{q}/tables')
        # Full unrounded continuation state, with the equation/geometry configuration.
        np.savez_compressed(ROOT/f'q{q}/final_state.npz',time_s=cfg['end_time'],faces=result['faces'],
            centers=result['centers'],volumes=result['volumes'],temperature_K=result['final_T'],
            moisture=result['final_C'],environment=env,config_json=json.dumps(result['metadata']))
        np.savetxt(ROOT/f'q{q}/tables/solver_diagnostics.csv',np.column_stack((result['times'][1:],result['logs'],result['water_balance'])),
            delimiter=',',header='time_s,mean_C,loss_per_initial_dry_mass,steps,iterations,rejections,min_dt,max_dt,max_scaled_residual,damped_iterations,water_balance_residual',comments='')
    if plots:
        if q==1:
            figure1(env,ROOT/'q1/figures');figure2(ROOT/'q1/figures');figure3(result,ROOT/'q1/figures')
        else:figure4(result,env,ROOT/'q2/figures')
    print(f'Question {q} complete; {result["metadata"]["steps"]} accepted steps',flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description='第一、二问求解与交付')
    p.add_argument('--question',choices=['1','2','all'],default='all')
    p.add_argument('--recompute',action='store_true',help='忽略已保存的正式计算，重新求解')
    p.add_argument('--no-plots',action='store_true');p.add_argument('--compute-only',action='store_true')
    p.add_argument('--validate',action='store_true',help='重新运行严格空间/时间加密与结果检查')
    args=p.parse_args()
    for q in ([1,2] if args.question=='all' else [int(args.question)]):
        run_question(q,recompute=args.recompute,plots=not(args.no_plots or args.compute_only),export=not args.compute_only)
    if args.validate:
        from validation.strict_batch import main as strict_main
        from validation.result_checks import core_checks,check_result
        strict_main();core_checks()
        for q in [1,2]:check_result(q,load_solution(ROOT/f'q{q}/solution.npz'),ROOT/f'submission_results/result{q}.xlsx')
    report=ROOT/'validation/strict_convergence.json'
    if report.exists() and not args.compute_only:
        convergence_figure(json.loads(report.read_text(encoding='utf-8')),ROOT/'validation/figures')
if __name__=='__main__':main()
