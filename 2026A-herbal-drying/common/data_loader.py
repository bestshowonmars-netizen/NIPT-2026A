from pathlib import Path
import numpy as np
from openpyxl import load_workbook

def load_environment(path):
    wb = load_workbook(Path(path), read_only=True, data_only=True)
    rows = list(wb.worksheets[0].values)
    wb.close()
    if tuple(rows[0]) != ("时间", "温度", "水分浓度"):
        raise ValueError("附件1表头不匹配")
    data = np.array(rows[1:], dtype=float)
    if data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError("环境数据含空值或非有限数")
    if data[0, 0] != 0 or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("环境时间必须从0开始且严格递增")
    data[:, 1] += 273.15
    return np.ascontiguousarray(data)
