"""Question-four settings for an independent solve from the supplied initial state."""

CONFIG = {
    "question": 4,
    "model_question": 4,
    "material_model": "appendix4",
    "radius": 0.02,
    "h": 25.0,
    "hm": 8e-7,
    "initial_T": 301.15,
    "initial_C": 2.55,
    "start_time": 0.0,
    "threshold": 0.15,
    "output_interval": 60.0,
    "sample_count": 101,
    "dt_max": 1.0,
    "dt_scale": 1.0/768.0,
    "time_growth": 600.0,
    "event_dt": 0.05,
    "epsilon_t": 0.01,
    "atol": 1e-10,
    "rtol": 1e-11,
    "max_iterations": 60,
    "shrinking": True,
    "initial_end_limit": 72.0*3600.0,
    "hard_end_limit": 30.0*24.0*3600.0,
    "horizon_growth": 2.0,
}


def make_config(dt_max=1.0, dt_scale=1.0/768.0, event_dt=0.05,
                event_tol=0.01, shrinking=True, end_limit=72.0*3600.0,
                hard_end_limit=30.0*24.0*3600.0, atol=None, rtol=None,
                sample_count=101, time_growth=600.0):
    config = dict(CONFIG)
    config.update(dt_max=float(dt_max), dt_scale=float(dt_scale),
                  time_growth=float(time_growth),
                  event_dt=min(float(dt_max), float(event_dt)),
                  epsilon_t=float(event_tol), shrinking=bool(shrinking),
                  initial_end_limit=float(end_limit), hard_end_limit=float(hard_end_limit))
    if atol is not None:
        config["atol"] = float(atol)
    if rtol is not None:
        config["rtol"] = float(rtol)
    if int(sample_count) != sample_count or sample_count < 21 or (int(sample_count)-1) % 20:
        raise ValueError("Sampling must include all 21 required physical and reference locations")
    config["sample_count"] = int(sample_count)
    for key in ("dt_max", "dt_scale", "event_dt", "epsilon_t", "atol", "rtol"):
        if not 0.0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if not 0.0 <= config["time_growth"] < float("inf"):
        raise ValueError("time_growth must be finite and nonnegative")
    if not 0.0 < config["initial_end_limit"] <= config["hard_end_limit"] < float("inf"):
        raise ValueError("Require 0 < end_limit <= hard_end_limit, in absolute seconds")
    return config
