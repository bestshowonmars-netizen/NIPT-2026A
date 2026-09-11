"""Run all four questions and their verified result exports."""
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
    if q==4:
        from q4.solve_q4 import solve_q4
        from q4.analysis import analyze
        selected_path=ROOT/'validation/q4_selected_config.json'
        if not selected_path.exists():
            raise RuntimeError('Run the Q4 refinement study before selecting formal settings')
        selected=json.loads(selected_path.read_text(encoding='utf-8'))
        def selected_solve(spec,shrinking):
            return solve_q4(cells=cells or spec['cells'],
                dt_max=dt if dt is not None else spec['dt_max'],
                dt_scale=spec['dt_scale'],event_dt=spec['event_dt'],
                event_tol=spec['event_tol'],shrinking=shrinking,recompute=recompute)
        result=selected_solve(selected,True)
        control=selected_solve(selected.get('fixed_control',selected),False)
        if not result['metadata']['completed'] or not control['metadata']['completed']:
            raise RuntimeError('Q4 and its appendix-4 fixed control require actual drying endpoints')
        if export:
            analyze(result,ROOT/'q4',plots=plots,
                comparison_solutions={'q3':load_solution(ROOT/'q3/solution.npz'),'a4_fixed':control})
        elif plots:
            from q4.plotting import figure6,figure7
            figure6(result['radius_data'],result['radius_slopes'],ROOT/'q4/figures')
            figure7(result,ROOT/'q4/figures',{'q3':load_solution(ROOT/'q3/solution.npz'),'a4_fixed':control})
        print(f"Question 4 complete: {result['metadata']['finish_time_h']:.10f} h",flush=True)
        return result
    if q==3:
        from q3.solve_q3 import solve_q3
        from q3.analysis import analyze
        selected_path=ROOT/'validation/q3_selected_config.json'
        if not selected_path.exists():
            raise RuntimeError('Run the Q3 convergence study before selecting a formal configuration')
        selected=json.loads(selected_path.read_text(encoding='utf-8'))
        result=None
        cached_path=ROOT/'q3/solution.npz'
        if not recompute and cells is None and dt is None and cached_path.exists():
            # Historical Q3 fingerprints contain the original absolute directory.
            # Verify the exact sources, arrays and refinement evidence in this
            # checkout before reusing the same result after a clone or relocation.
            from validation.q3_acceptance import validate_selected
            candidate=load_solution(cached_path)
            verification=validate_selected(candidate)
            if not verification['passed']:
                raise RuntimeError('The saved Q3 result failed its published acceptance checks')
            result=candidate
            print('Q3 cache: accepted source hashes and arrays verified in this checkout',flush=True)
        if result is None:
            result=solve_q3(cells=cells or selected['cells'],
                dt_max=dt if dt is not None else selected['dt_max'],
                event_dt=selected['event_dt'],event_tol=selected['event_tol'],
                recompute=recompute)
        if not result['metadata']['completed']:
            raise RuntimeError('The search limit was reached without a valid Q3 endpoint')
        if export:
            analyze(result,ROOT/'q3',plots=plots)
        elif plots:
            from q3.plotting import figure5
            figure5(result,ROOT/'q3/figures')
        print(f"Question 3 complete: {result['metadata']['finish_time_h']:.10f} h",flush=True)
        return result
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
    p=argparse.ArgumentParser(description='第一至四问求解与交付')
    p.add_argument('--question',choices=['1','2','3','4','all'],default='all')
    p.add_argument('--recompute',action='store_true',help='忽略已保存的正式计算，重新求解')
    p.add_argument('--no-plots',action='store_true');p.add_argument('--compute-only',action='store_true')
    p.add_argument('--validate',action='store_true',help='检查结果、续算一致性和已有收敛记录；第一、二问另运行严格加密矩阵')
    args=p.parse_args()
    questions=[1,2,3,4] if args.question=='all' else [int(args.question)]
    for q in questions:
        run_question(q,recompute=args.recompute,plots=not(args.no_plots or args.compute_only),export=not args.compute_only)
    if args.validate and any(q in [1,2] for q in questions):
        from validation.strict_batch import main as strict_main
        from validation.result_checks import core_checks,check_result
        strict_main();core_checks()
        for q in [1,2]:check_result(q,load_solution(ROOT/f'q{q}/solution.npz'),ROOT/f'submission_results/result{q}.xlsx')
    if args.validate and 3 in questions:
        from validation.q3_checks import check_solution,check_deliverables,core_restart_checks
        from validation.q3_acceptance import validate_selected
        q3solution=load_solution(ROOT/'q3/solution.npz')
        acceptance=validate_selected(q3solution)
        (ROOT/'validation/q3_acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2),encoding='utf-8')
        if not acceptance['passed']:
            raise RuntimeError('The selected Q3 configuration failed a required refinement comparison')
        q3report={'core':core_restart_checks(),
            'selected_refinements_passed':acceptance['passed'],
            'solution':check_solution(q3solution,load_solution(ROOT/'q2/solution.npz'),require_dense=True),
            'deliverables':check_deliverables(q3solution,ROOT/'submission_results/result3.xlsx',
                ROOT/'q3/tables/table5.csv',ROOT/'q3/endpoint.json')}
        (ROOT/'validation/q3_checks.json').write_text(json.dumps(q3report,ensure_ascii=False,indent=2),encoding='utf-8')
        from q3.paper_section import write_section
        write_section()
    if args.validate and 4 in questions:
        from validation.q4_checks import check_solution,check_deliverables,core_checks
        from validation.q4_acceptance import validate_selected
        result=load_solution(ROOT/'q4/solution.npz')
        acceptance=validate_selected()
        if not acceptance['passed']:
            raise RuntimeError('Q4 formal settings lack passing, current refinement evidence')
        report={'core':core_checks(), 'selected_refinements_passed':True,
            'solution':check_solution(result),
            'fixed_control':check_solution(load_solution(ROOT/'q4/fixed_control.npz')),
            'deliverables':check_deliverables(result,ROOT/'submission_results/result4.xlsx',
                ROOT/'q4/tables/table6.csv',ROOT/'q4/endpoint.json')}
        (ROOT/'validation/q4_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report=ROOT/'validation/strict_convergence.json'
    if report.exists() and not args.compute_only and any(q in [1,2] for q in questions):
        convergence_figure(json.loads(report.read_text(encoding='utf-8')),ROOT/'validation/figures')
if __name__=='__main__':main()
