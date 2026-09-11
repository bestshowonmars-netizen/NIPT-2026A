from pathlib import Path
import csv,json
import numpy as np

def write_tables_and_note(solution,question,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    times=[100,300,600,900,1200,1500,1800] if question==1 else [1800,3600,5400,7200,9000,10800]
    positions=[0,.5,1,1.5,2]
    indices=[int(round(x/2*(len(solution["radii"])-1))) for x in positions]
    units="s" if question==1 else "h"
    parts=[]
    for key,label,table_number in [("temperature_K","温度/°C",1 if question==1 else 3),
                                   ("moisture","干基含水率/(kg/kg)",2 if question==1 else 4)]:
        header=[f"时间/{units}"]+[f"r={r:g} cm" for r in positions]
        rows=[]
        for t in times:
            values=solution[key][t,indices]-(273.15 if key=="temperature_K" else 0)
            rows.append([t if question==1 else t/3600]+[f"{v:.4f}" for v in values])
        with (folder/f"table{table_number}.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
        md=f"## 表{table_number} {label}\n\n| "+" | ".join(header)+" |\n|"+"---|"*6+"\n"
        md+="\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)+"\n"
        (folder/f"table{table_number}.md").write_text(md,encoding="utf-8");parts.append(md)
    (folder/"题目规定表格.md").write_text("\n".join(parts),encoding="utf-8")
    T=solution["temperature_K"][-1]-273.15;C=solution["moisture"][-1]
    meta=solution["metadata"]
    if question==1:
        method="将药材视为固定半径的一维径向圆柱体，对同心壳层建立热量与水分通量平衡；热物性使用附录2常数，水分扩散系数按各壳层当地干基含水率更新。附件1经分段线性插值作为表面对流驱动，采用有限体积、全隐式时间推进及Picard迭代，计算前1800秒。温度与水分分别求解，中心按对称条件、表面按对流边界重构。"
    else:
        method="保留固定半径、一维径向几何与表面对流边界，从题设均匀初态重新计算前3小时；所有物性统一采用附录3，温度使用开尔文进入扩散系数公式。每个全隐式时间步通过Picard迭代交替更新温度、含水率和当地物性，变系数保留在有限体积通量内部。表面对流系数约定沿用附录2，不与第一问结果拼接。"
    mean=float(solution["logs"][-1,0]);loss=float(solution["logs"][:,1].sum())
    explanation=(f"在{int(meta['end_time'])}秒时，中心与表面温度分别为{T[0]:.4f}、{T[-1]:.4f} °C，"
                 f"中心与表面干基含水率分别为{C[0]:.4f}、{C[-1]:.4f} kg/kg。"
                 f"此时表面比中心高{T[-1]-T[0]:.4f} °C，中心比表面高{C[0]-C[-1]:.4f} kg/kg，"
                 "说明该时刻的外部加热与表面失水尚存在径向响应差异。"
                 f"全域按干物质质量加权的平均含水率为{mean:.6f} kg/kg，单位初始干物质质量累计失水为{loss:.6f} kg/kg。"
                 "这些是当前有效模型的预测，不是内部温湿场实验验证，也不用于推断最终干燥时长。")
    caveat="两问均将环境水分指标作为药材侧等效驱动，不将空气与固体的含水质量基准等同；主模型不显式加入潜热，采用一维径向近似。四位小数的稳定性及其边界以validation中的独立加密报告为准。"
    (folder.parent/"方法草稿与结果解释.md").write_text("# 问题"+str(question)+"：给论文手的方法草稿与结果解释\n\n"+method+"\n\n"+explanation+"\n\n"+caveat+"\n",encoding="utf-8")
