"""Package a checked Q4 project and the separate paper-facing materials."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT=bootstrap.ROOT


def build(destination):
    from validation.q4_acceptance import validate_selected
    acceptance=validate_selected()
    assert acceptance['passed']
    checks=json.loads((ROOT/'validation/q4_checks.json').read_text(encoding='utf-8'))
    from validation.q4_reproduction import verify
    reproduction=verify()
    assert checks['solution']['passed'] and checks['fixed_control']['passed']
    assert checks['deliverables']['passed'] and reproduction['passed']
    assert checks['core']['passed']
    for name,digest in reproduction['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    from common.solver import load_solution
    model=load_solution(ROOT/'q4/solution.npz')
    control=load_solution(ROOT/'q4/fixed_control.npz')
    from validation.q4_checks import check_deliverables
    fresh_deliverables=check_deliverables(model,ROOT/'submission_results/result4.xlsx',
        ROOT/'q4/tables/table6.csv',ROOT/'q4/endpoint.json')
    assert fresh_deliverables['passed']
    required=['q4/tables/table6.md','q4/论文正文_问题4.tex',
              'validation/问题4精度验证报告.md','validation/figures/q4_convergence.pdf']
    required += [f'q4/figures/{stem}.{extension}' for stem in
                 ('fig6_q4_radius_mapping','fig7_q4_shrinking_comparison','fig8_combined_convergence')
                 for extension in ('pdf','svg','png')]
    for name in required:
        assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0,('Missing deliverable',name)
    t,tf=model['metadata']['finish_time_s'],control['metadata']['finish_time_s']
    statement=f'''# 第四问交付说明

第四问已从题设均匀初态计算至全域严格达标。内部联合求解温度与含水率，附录4物性及附件2半径历程从初始时刻共同生效。

\\[
t_4={t:.11f}\\,\\mathrm{{s}}\\approx {t/3600:.4f}\\,\\mathrm{{h}},\\qquad
M_4(t_+)={model['max_C'][-1]:.17g}<0.15.
\\]

同物性固定半径对照为 {tf/3600:.3f} h。本模型下两者相差 {(tf-t)/3600:.3f} h，相对固定尺寸缩短 {(tf-t)/tf*100:.3f}%。第三问与第四问同时改变了物性与几何，不能将这两问之差全部归因于收缩。

直接使用 `submission_results/result4.xlsx`、`q4/tables/table6.md`、`q4/endpoint.json`、`q4/figures` 和 `q4/论文正文_问题4.tex`。图6展示半径输入与材料坐标；图7展示实际收缩区域的含水率和三组模型比较；图8汇总第三、四问的终点加密检查。图均提供PDF、SVG、300dpi PNG。

固定物理点只有在未舍入半径以内才输出，域外留空；W列是独立的移动表面值。正文表6含每6小时及精确终点；Excel只含每60秒的规则时刻。

网格、时间步、初始小步、Picard与局部事件验收见 `validation/问题4精度验证报告.md` 和对应JSON。完整独立运行的附录程序从原始初态重算，已与正式数值解及Excel逐格核对。局部事件括区、数值加密差异和物理模型假设的含义分别说明；数值核查不等同于内部场实验验证。

在完整工程根目录运行：

```powershell
python -B run_all.py --question 4 --validate
```

重新计算增加 `--recompute`。独立附录复现执行 `python -B validation/q4_reproduction.py --run`。依赖使用工程原有 `requirements.txt`，无需新的绘图或商业软件。

当前工程已完整实现第一至四问，第一至三问数值内核和正式结果已保留。此说明记录第四问本地归档内容；打包器不执行GitHub推送，清单中的 `remote_publication_performed=false` 仅表示该次打包操作没有发布远程版本。当前仓库发布状态及对应版本以Git历史和远程分支为准。压缩包内附逐文件SHA256清单。
'''
    (ROOT/'第四问交付说明.md').write_text(statement,encoding='utf-8')
    destination=Path(destination).resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError('Delivery output must be outside the project')
    destination.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(p.startswith('.') or p=='__pycache__' or p.startswith('preview') for p in relative.parts):continue
        if path.suffix.lower() in ('.pyc','.pyo','.pyd','.nbc','.nbi'):continue
        if 'pilot' in path.name or 'smoke' in path.name:continue
        files.append((relative,path,{'path':relative.as_posix(),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()
    receipts=[]
    for name,chosen in (
        ('问题4完整工程.zip',files),
        ('问题4论文材料.zip',[f for f in files if
            (f[0].parts[0]=='q4' and (f[0].parts[1] in ('figures','tables') or f[0].suffix in ('.tex',)))
            or (f[0].parent.as_posix()=='validation/figures' and f[0].stem=='q4_convergence')
            or f[0].as_posix() in ('submission_results/result4.xlsx','q4/endpoint.json',
                '第四问交付说明.md','validation/问题4精度验证报告.md','validation/q4_selected_config.json')]),
    ):
        archive=destination/name
        manifest={'project':ROOT.name,'git_base_or_head':version,'remote_publication_performed':False,
            'finish_time_s':t,'fixed_control_finish_time_s':tf,'acceptance_passed':True,
            'standalone_reproduction_passed':True,'files':[r for _,_,r in chosen]}
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for relative,path,_ in chosen:z.write(path,f'{ROOT.name}/{relative.as_posix()}')
            z.writestr(f'{ROOT.name}/交付文件校验清单.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for relative,_,record in chosen:
                assert hashlib.sha256(z.read(f'{ROOT.name}/{relative.as_posix()}')).hexdigest()==record['sha256']
        receipts.append({'archive':str(archive),'bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(chosen),
            'all_archive_members_verified':True})
    (destination/'交付包校验.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipts,ensure_ascii=False,indent=2))
    return receipts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    build(p.parse_args().output)
