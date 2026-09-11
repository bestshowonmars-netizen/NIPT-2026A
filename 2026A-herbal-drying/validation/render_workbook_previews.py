from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
from common.plotting import setup
import matplotlib.pyplot as plt
from openpyxl import load_workbook

def render(q):
    wb=load_workbook(bootstrap.ROOT/f'submission_results/result{q}.xlsx',read_only=True,data_only=True)
    setup()
    for sheet in wb:
        all_rows=list(sheet.values)
        selected=[1,2,3,4,len(all_rows)-2,len(all_rows)-1]
        head=list(all_rows[0][:8]);head[0]='时间 / s\n距离 / cm'
        rows=[]
        for idx in selected:
            row=all_rows[idx][:8]
            rows.append([str(row[0])]+[f'{v:.4f}' for v in row[1:]])
        fig,ax=plt.subplots(figsize=(8.2,2.9));ax.axis('off')
        table=ax.table(cellText=rows,colLabels=head,cellLoc='center',loc='center',colWidths=[.18]+[.115]*7)
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#C8CDD1');cell.set_linewidth(.5)
            if r==0:cell.set_facecolor('#EAF0F4');cell.set_height(.19)
        ax.set_title(f'result{q}.xlsx — {sheet.title}：A:H 首行与末行抽检',fontsize=10,pad=10)
        folder=bootstrap.ROOT/'validation/previews';folder.mkdir(exist_ok=True)
        fig.savefig(folder/f'result{q}_{sheet.title}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    wb.close()
if __name__=='__main__':
    for q in [1,2]:
        if (bootstrap.ROOT/f'submission_results/result{q}.xlsx').exists():render(q)
