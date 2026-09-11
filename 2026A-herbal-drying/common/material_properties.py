import numpy as np
from numba import njit

@njit(cache=True)
def properties(T, C, question):
    if question == 1:
        return 820.0 * 2600.0, 0.36, 7e-9 * np.exp(-0.89 / C)
    if question == 2:
        a = (650.0 + 128.0 * C) * (1450.0 + 2736.0 * C / (C + 1.0))
        k = 0.21 + 0.38 * C / (C + 1.0)
        D = 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T)
        return a, k, D
    raise ValueError("Only questions 1 and 2 are implemented")
