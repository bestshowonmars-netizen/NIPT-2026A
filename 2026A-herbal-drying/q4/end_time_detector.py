"""Strict Q4 endpoint bracketing by actual integration from an unchanged anchor."""
import numpy as np
from q4.kernel import advance_interval, global_maximum


def refine_endpoint(T, C, t_start, t_stop, environment, faces, centers, volumes,
                    config, radius_data, radius_slopes):
    """Recompute each candidate from the same full-grid last-undried state.

    No state or crossing time is interpolated. A coarse crossing that vanishes
    under the event step returns a nonterminal re-integrated upper state.
    Only the accepted trajectory contributes to cumulative water loss.
    """
    h, hm = config["h"], config["hm"]
    threshold, shrinking = config["threshold"], config["shrinking"]
    m0, r0, _, _ = global_maximum(T, C, t_start, environment, faces, centers,
        h, hm, radius_data, radius_slopes, shrinking)
    if not np.isfinite(m0) or m0 < threshold:
        raise ValueError("Endpoint refinement needs a finite undried anchor with g >= 0")
    if not t_stop > t_start:
        raise ValueError("Endpoint interval must have positive length")
    trace = [[float(t_start), float(m0), float(r0), float(m0-threshold)]]
    work_steps = work_iterations = 0

    def evaluate(t):
        nonlocal work_steps, work_iterations
        tn, cn, diag = advance_interval(T, C, float(t_start), float(t),
            config["event_dt"], environment, faces, centers, volumes,
            h, hm, config["atol"], config["rtol"], config["max_iterations"],
            radius_data, radius_slopes, shrinking, config["dt_scale"], config["time_growth"])
        m, r, cc, cs = global_maximum(tn, cn, t, environment, faces, centers,
            h, hm, radius_data, radius_slopes, shrinking)
        if not np.isfinite(m) or not np.isfinite(diag).all():
            raise RuntimeError("Nonfinite event state or diagnostics")
        trace.append([float(t), float(m), float(r), float(m-threshold)])
        work_steps += int(diag[2])
        work_iterations += int(diag[3])
        return {"T": tn, "C": cn, "diag": diag, "M": float(m), "r": float(r),
                "C_center": float(cc), "C_surface": float(cs), "time": float(t)}

    upper = evaluate(t_stop)
    lo, hi = float(t_start), float(t_stop)
    glo, ghi = float(m0-threshold), upper["M"]-threshold
    lower_T, lower_C, lower_diag = T.copy(), C.copy(), np.zeros(9)
    if ghi >= 0.0:
        return {**upper, "crossed": False, "trace": np.asarray(trace), "bracket": None,
                "work_steps": work_steps, "work_iterations": work_iterations}
    while hi-lo > config["epsilon_t"]:
        mid = lo+(hi-lo)*0.5
        if mid == lo or mid == hi:
            raise RuntimeError("Endpoint tolerance is below floating-point time resolution")
        candidate = evaluate(mid)
        g = candidate["M"]-threshold
        if g < 0.0:
            hi, ghi, upper = mid, g, candidate
        else:
            lo, glo = mid, g
            lower_T, lower_C, lower_diag = candidate["T"], candidate["C"], candidate["diag"]
    if not (glo >= 0.0 and ghi < 0.0):
        raise RuntimeError("A strict endpoint sign bracket was not established")
    return {**upper, "crossed": True, "trace": np.asarray(trace),
            "bracket": {"t_minus": lo, "t_plus": hi, "g_minus": glo,
                        "g_plus": ghi, "width": hi-lo},
            "lower_T": lower_T, "lower_C": lower_C, "lower_diag": lower_diag,
            "work_steps": work_steps, "work_iterations": work_iterations}
