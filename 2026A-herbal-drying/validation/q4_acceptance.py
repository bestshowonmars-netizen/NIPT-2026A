"""Recheck selected Q4 settings against current, unrounded refinement cases."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import hashlib
import json
import numpy as np
from common.solver import load_solution
from q4.convergence_test import difference
from validation.q4_checks import check_solution

ROOT=bootstrap.ROOT


def validate_selected():
    selected=json.loads((ROOT/'validation/q4_selected_config.json').read_text(encoding='utf-8'))
    cached={}
    def case(path):
        relative=str(path).replace('\\','/')
        if relative not in cached:
            data=load_solution(ROOT/relative)
            checks=check_solution(data)
            cached[relative]=(data,checks)
        return cached[relative][0]
    formal={}
    for scope,path,spec in (('shrinking','q4/solution.npz',selected),
                            ('fixed','q4/fixed_control.npz',selected['fixed_control'])):
        output=case(path);source=case(spec['formal_case'])
        for key in ('cells','dt_max','dt_scale','event_dt'):
            assert output['metadata'][key]==spec[key],(scope,key)
        assert output['metadata']['epsilon_t']==spec['event_tol']
        assert output['metadata']['shrinking']==(scope=='shrinking')
        assert output['metadata']['cache_fingerprint']==source['metadata']['cache_fingerprint']
        for key in ('times','final_T','final_C','max_C','radius_m'):
            assert np.array_equal(output[key],source[key]),(scope,key,'promotion changed numerical values')
        formal[scope]={'path':path,'formal_case':str(spec['formal_case']).replace('\\','/'),
            'finish_time_s':output['metadata']['finish_time_s']}
    results=[];covered=set();digests={}
    for requirement in selected['required_comparisons']:
        filename=requirement['report'];index=int(requirement.get('index',0))
        report=json.loads((ROOT/filename).read_text(encoding='utf-8'))
        saved=report['comparisons'][index]
        assert saved['passed'],(filename,index,'saved comparison failed')
        scope=requirement['scope'];kind=saved['kind']
        paths=saved.get('case_paths',{'coarse':saved.get('coarse_path'),'fine':saved.get('fine_path')})
        left,right=case(paths['coarse']),case(paths['fine'])
        assert left['metadata']['shrinking']==right['metadata']['shrinking']==(scope=='shrinking')
        allowed={'time':{'dt_max','event_dt'},'startup':{'dt_scale'},
                 'space':{'cells'},'picard_tolerance':{'atol','rtol'},
                 'event_tolerance':{'epsilon_t'},'event_integration_step':{'event_dt'}}
        normalized=kind.removeprefix('fixed_')
        family='space' if normalized.startswith('space') else normalized
        assert family in allowed,(filename,kind)
        spec=selected if scope=='shrinking' else selected['fixed_control']
        lm,rm=left['metadata'],right['metadata']
        if family=='time':
            assert lm['cells']==rm['cells']==spec['cells'],'Time comparison must use the selected grid'
            assert lm['dt_max']>rm['dt_max']==spec['dt_max'],'The selected time step must be the refined member'
            assert lm['dt_scale']==rm['dt_scale']==spec['dt_scale']
        elif family=='space':
            assert lm['cells']==spec['cells']<rm['cells'],'Compare the selected grid with a genuinely finer grid'
            assert lm['dt_scale']==rm['dt_scale']==spec['dt_scale']
        elif family=='startup':
            assert lm['cells']==rm['cells']==spec['cells'],'Startup evidence must use the selected grid'
            assert lm['dt_scale']>rm['dt_scale'] and spec['dt_scale'] in (lm['dt_scale'],rm['dt_scale'])
        elif family=='picard_tolerance':
            assert lm['cells']==rm['cells']==spec['cells'],'Picard evidence must use the selected grid'
            assert lm['dt_scale']==rm['dt_scale']==spec['dt_scale']
            fm=case(formal[scope]['path'])['metadata']
            assert (fm['atol'],fm['rtol']) in ((lm['atol'],lm['rtol']),(rm['atol'],rm['rtol']))
        if family in ('event_tolerance','event_integration_step'):
            expected={(ROOT/formal[scope]['formal_case']).resolve(),
                      (ROOT/formal[scope]['path']).resolve()}
            parents=[s['metadata'].get('validation_parent_path') for s in (left,right)]
            assert any(parents),'Local event evidence must retain its full-trajectory parent'
            for parent in parents:
                if parent is not None:
                    assert (ROOT/str(parent).replace('\\','/')).resolve() in expected,('Event parent is not the selected formal trajectory',parent)
        keys=('cells','dt_max','dt_scale','time_growth','event_dt','epsilon_t','shrinking','atol','rtol')
        changed={k for k in keys if left['metadata'][k]!=right['metadata'][k]}
        assert changed and changed<=allowed[family],(filename,changed,'comparison mixes numerical factors')
        fresh=difference(left,right,kind,paths['coarse'],paths['fine'])
        assert fresh['passed'],(filename,index,'current unrounded comparison failed')
        assert fresh['targets']==saved['targets']
        assert abs(fresh['C']['maximum']-saved['C']['maximum'])<=1e-14
        assert abs(fresh['finish_time_s']['difference']-saved['finish_time_s']['difference'])<=1e-10
        results.append({'scope':scope,'report':filename,'index':index,
                        'coarse':saved.get('coarse',Path(paths['coarse']).stem),
                        'fine':saved.get('fine',Path(paths['fine']).stem),
                        'selected_grid_binding_verified':True,
                        'formal_dt_max':spec['dt_max'],
                        'comparison_dt_max':[lm['dt_max'],rm['dt_max']],**fresh})
        covered.add((scope,family))
        digests[filename]=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    required={('shrinking',k) for k in ('time','startup','space','picard_tolerance',
                                      'event_tolerance','event_integration_step')}
    required|={('fixed','time'),('fixed','space')}
    assert required<=covered,('Missing required comparisons',required-covered)
    for name in cached:digests[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    result={'passed':True,'formal':formal,'comparisons':results,'source_hashes':digests,
            'checked_case_count':len(cached),'required_checks_covered':[list(x) for x in sorted(covered)],
            'note':'Actual numerical differences on the stated cases, not a strict total error bound'}
    (ROOT/'validation/q4_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    report=validate_selected()
    print(json.dumps({'passed':report['passed'],'formal':report['formal'],
                      'checked_case_count':report['checked_case_count']},ensure_ascii=False,indent=2))
