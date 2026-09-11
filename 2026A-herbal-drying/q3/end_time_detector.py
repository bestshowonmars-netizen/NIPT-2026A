"""Locate a strict moisture crossing by re-integrating saved full states."""
import numpy as np
from common.continuation import advance_interval, global_maximum


def refine_endpoint(T, C, t_start, t_stop, environment, faces, centers, volumes, config):
    """Evaluate every candidate from the same last-undried state, without interpolation.

    The returned diagnostic/loss belongs only to the accepted upper trajectory;
    work spent on discarded bisection candidates is reported separately.
    A coarse crossing can disappear under smaller steps. In that case `crossed`
    is false and the returned state is the re-integrated nonterminal upper state.
    """
    threshold = config["threshold"]
    h, hm = config["h"], config["hm"]
    m0, r0, cc0, cs0 = global_maximum(T, C, t_start, environment, faces, centers, h, hm)
    g0 = float(m0 - threshold)
    if g0 < 0.0:
        raise ValueError("Endpoint refinement requires an undried anchor, g >= 0")
    if not t_stop > t_start:
        raise ValueError("Endpoint interval must have positive length")
    trace = [[float(t_start), float(m0), float(r0), g0]]
    work_steps = 0
    work_iterations = 0

    def evaluate(t):
        nonlocal work_steps, work_iterations
        tn, cn, diag = advance_interval(
            T, C, float(t_start), float(t), config["event_dt"], environment,
            faces, centers, volumes, h, hm, config["atol"], config["rtol"],
            config["max_iterations"])
        m, r, cc, cs = global_maximum(tn, cn, t, environment, faces, centers, h, hm)
        trace.append([float(t), float(m), float(r), float(m - threshold)])
        work_steps += int(diag[2])
        work_iterations += int(diag[3])
        return {"T": tn, "C": cn, "diag": diag, "M": float(m), "r": float(r),
                "C_center": float(cc), "C_surface": float(cs), "time": float(t)}

    upper = evaluate(t_stop)
    lower_T, lower_C = T.copy(), C.copy()
    lower_diag = np.zeros(9)
    lo, hi = float(t_start), float(t_stop)
    glo, ghi = g0, upper["M"] - threshold
    if ghi >= 0.0:
        return {**upper, "crossed": False, "trace": np.asarray(trace),
                "bracket": None, "work_steps": work_steps, "work_iterations": work_iterations}
    while hi - lo > config["epsilon_t"]:
        mid = lo + (hi - lo) * 0.5
        if mid == lo or mid == hi:
            raise RuntimeError("Endpoint tolerance is below floating-point time resolution")
        candidate = evaluate(mid)
        g = candidate["M"] - threshold
        if g < 0.0:
            hi, ghi, upper = mid, g, candidate
        else:
            lo, glo = mid, g
            lower_T, lower_C, lower_diag = candidate["T"], candidate["C"], candidate["diag"]
    if not (glo >= 0.0 and ghi < 0.0):
        raise RuntimeError("A strict endpoint sign bracket was not established")
    return {**upper, "crossed": True, "trace": np.asarray(trace),
            "bracket": {"t_minus": lo, "t_plus": hi, "g_minus": glo,
                        "g_plus": ghi, "width": hi - lo},
            "lower_T": lower_T, "lower_C": lower_C, "lower_diag": lower_diag,
            "work_steps": work_steps, "work_iterations": work_iterations}
