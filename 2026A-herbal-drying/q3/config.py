"""Numerical and output settings; physical properties remain question two."""
from q2.config import CONFIG as Q2_CONFIG

CONFIG = {
    **Q2_CONFIG,
    "start_time": 10800.0,
    "threshold": 0.15,
    "output_interval": 60.0,
    "sample_count": 101,
    "dt_max": 1.0,
    "event_dt": 0.05,
    "epsilon_t": 0.01,
    "initial_end_limit": 72.0 * 3600.0,
    "hard_end_limit": 30.0 * 24.0 * 3600.0,
    "horizon_growth": 2.0,
}
# end_time in Q2_CONFIG describes the supplied prefix, not a Q3 stopping time.
CONFIG.pop("end_time")


def make_config(dt_max=1.0, event_dt=0.05, event_tol=0.01,
                end_limit=72.0 * 3600.0, hard_end_limit=30.0 * 24.0 * 3600.0,
                atol=None, rtol=None):
    cfg = dict(CONFIG)
    cfg.update(dt_max=float(dt_max), event_dt=min(float(dt_max), float(event_dt)),
               epsilon_t=float(event_tol), initial_end_limit=float(end_limit),
               hard_end_limit=float(hard_end_limit))
    if atol is not None:
        cfg["atol"] = float(atol)
    if rtol is not None:
        cfg["rtol"] = float(rtol)
    for key in ("dt_max", "event_dt", "epsilon_t", "atol", "rtol"):
        if not 0.0 < cfg[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if not cfg["start_time"] < cfg["initial_end_limit"] <= cfg["hard_end_limit"] < float("inf"):
        raise ValueError("Require start_time < end_limit <= hard_end_limit, in absolute seconds")
    return cfg
