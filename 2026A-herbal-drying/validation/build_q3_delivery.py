"""Package the verified project snapshot, excluding runtime caches and previews."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT = bootstrap.ROOT


def build(destination):
    acceptance = json.loads((ROOT/"validation/q3_acceptance.json").read_text(encoding="utf-8"))
    reproduction = json.loads((ROOT/"validation/q3_reproduction.json").read_text(encoding="utf-8"))
    checks = json.loads((ROOT/"validation/q3_checks.json").read_text(encoding="utf-8"))
    assert acceptance["passed"] and reproduction["passed"]
    assert checks["solution"]["passed"] and checks["deliverables"]["passed"]
    endpoint = json.loads((ROOT/"q3/endpoint.json").read_text(encoding="utf-8"))
    summary = f"""# 第三问交付说明

第三问已沿用第二问附录3模型，从3 h完整960单元状态按绝对时间续算至全域严格达标。内部持续计算温度与含水率，正式续算最大步长为1/32 s。

\\[
t_{{\\mathrm{{finish}}}}={endpoint['finish_time_s']:.11f}\\ \\mathrm{{s}}
\\approx {endpoint['finish_time_h']:.4f}\\ \\mathrm{{h}},\\qquad
M_h(t_+)={endpoint['max_C']:.17g}<0.15.
\\]

最后达标位置为中心。事件夹逼区间为 [{endpoint['bracket']['t_minus']:.11f}, {endpoint['bracket']['t_plus']:.11f}] s，宽度为 {endpoint['bracket']['width']:.11f} s；该宽度是局部事件定位分辨率。

已通过8项正式选档比较、完整状态与水量检查、Excel逐格核对和同算法附录全过程复现。网格与时间差是所列数值档次之间的实测差，不是未知精确解的严格误差界。边界延续对照另列于验证报告；正式方案保持原先约定的4 h后50°C、0.0500 kg/kg。

## 直接使用的文件

- `submission_results/result3.xlsx`：每60 s、每0.1 cm的含水率，3448行、72408个浓度值。
- `q3/tables/table5.md`：每6 h和精确终点的正文表；`q3/endpoint.json`：未舍入终点、夹逼区间及完整网格状态。
- `q3/figures/fig5_q3_drying_endpoint.pdf`：阈值交叉与含水率时空图；`fig8_q3_convergence.pdf`：收敛补图。两图均另有SVG与300 dpi PNG。
- `q3/论文正文_问题3.tex`：LaTeX正文片段；`q3/数值验证配图.tex`：供全文数值验证部分插入的配图片段。
- `validation/问题3精度验证报告.md`：数值验证、边界假设对照和独立运行复现的完整记录。
- `paper_appendix_minimal`：可从原附件重新计算第二、三问的自包含附录程序。

## 运行

在本目录安装 `requirements.txt` 后执行：

```powershell
python -B run_all.py --question 3 --validate
```

更详细的重新计算与验证命令见 `q3/README.md`、`paper_appendix_minimal/README.md`。当前工程已完整实现第一至四问，本说明记录第三问交付内容；第四问另见其交付说明与 `q4/README.md`。数值解依赖已说明的有效传质、潜热及径向模型假设。

本说明所述归档由本地打包器生成；打包器不执行GitHub推送，清单中的 `remote_publication_performed=false` 仅表示该次打包操作没有发布远程版本。当前仓库发布状态及对应版本以Git历史和远程分支为准。压缩包保存实际文件快照，内附逐文件SHA256清单，不包含本机依赖缓存与临时预览。
"""
    (ROOT/"第三问交付说明.md").write_text(summary, encoding="utf-8")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_relative_to(ROOT):
        raise ValueError("Delivery output must be outside the project to avoid recursive inclusion")
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part.startswith(".") or part == "__pycache__" or part.startswith("preview") for part in relative.parts):
            continue
        if path.suffix.lower() in (".pyc", ".pyo", ".pyd", ".nbi", ".nbc"):
            continue
        if path.name.startswith("pilot_") or path.name.startswith("q3_pilot_"):
            continue
        data = path.read_bytes()
        files.append((relative, path, {"path": relative.as_posix(), "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}))
    manifest = {"project": ROOT.name, "git_base_or_head": version,
                "remote_publication_performed": False,
                "finish_time_s": endpoint["finish_time_s"],
                "acceptance_passed": True, "standalone_reproduction_passed": True,
                "files": [record for _, _, record in files]}
    archive = destination/"问题3完整工程.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for relative, path, record in files:
            stream.write(path, f"{ROOT.name}/{relative.as_posix()}")
        stream.writestr(f"{ROOT.name}/交付文件校验清单.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for relative, _, record in files:
            assert hashlib.sha256(stream.read(f"{ROOT.name}/{relative.as_posix()}")).hexdigest() == record["sha256"]
    receipt = {"archive": str(archive), "archive_bytes": archive.stat().st_size,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "included_files": len(files), "all_archive_members_verified": True,
               "finish_time_s": endpoint["finish_time_s"], "source_project": str(ROOT)}
    (destination/"交付包校验.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
