import numpy as np
from numba import njit

@njit(cache=True)
def environment_at(t, env):
    if t > env[-1, 0]:
        return 323.15, 0.0500
    return np.interp(t, env[:, 0], env[:, 1]), np.interp(t, env[:, 0], env[:, 2])
