"""Template-preserving numerical outputs; no solver calculations in Excel."""
from pathlib import Path
from copy import copy
from openpyxl import load_workbook

def export_result(solution,template_path,destination):
    wb=load_workbook(template_path)
    for sheet,key in [("温度","temperature_K"),("水分浓度","moisture")]:
        ws=wb[sheet]
        # Replace illustrative ellipses; source template file is never edited.
        for row in ws:
            for cell in row:
                if cell.coordinate != "A1": cell.value=None
        for j in range(21):ws.cell(1,j+2,round(j*.1,1))
        stride=(len(solution["radii"])-1)//20
        values=solution[key][:,::stride]
        for i in range(1,len(solution["times"])):
            ws.cell(i+1,1,int(solution["times"][i]))
            for j in range(21):
                value=float(values[i,j])-(273.15 if key=="temperature_K" else 0)
                cell=ws.cell(i+1,j+2,round(value,4));cell.number_format="0.0000"
        ws.freeze_panes="B2"
        ws.column_dimensions["A"].width=28
        from openpyxl.utils import get_column_letter
        for j in range(2,23):ws.column_dimensions[get_column_letter(j)].width=11
        ws.row_dimensions[1].height=32
        from openpyxl.styles import Alignment
        ws["A1"].alignment=Alignment(wrap_text=True,vertical="center")
    wb.properties.creator="";wb.properties.lastModifiedBy=""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    wb.save(destination)
