"""Matched-step sensitivity to the already-declared post-4h boundary assumption.

This control holds attachment 1's last values by appending the identical values
to the environment table; it still uses the existing environment_at function.
It is a model-assumption comparison, not a replacement for the formal solution.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import hashlib
import json
import time
import numpy as np
from common.solver import load_solution, save_solution
from common.data_loader import load_environment
from common.continuation import advance_interval, sample_fields, global_maximum
from q3.end_time_detector import refine_endpoint
from q3.convergence_test import difference

ROOT = bootstrap.ROOT


def run():
    prefix = load_solution(ROOT/"q2/solution.npz")
    env_original = load_environment(ROOT/"data/attachment1.xlsx")
    env = np.vstack((env_original, [30*24*3600., *env_original[-1, 1:]]))
    cfg = {**{k: prefix["metadata"][k] for k in ("radius", "h", "hm", "atol", "rtol",
                                               "max_iterations", "initial_T", "initial_C", "mesh", "cells")},
           "source_metadata": prefix["metadata"], "sample_count": 21,
           "question": 3, "model_question": 2,
           "start_time_s": 10800., "threshold": .15, "dt_max": 1.,
           "event_dt": .05, "epsilon_t": .01,
           "boundary_control": "hold last attachment values after 4h",
           "tail_temperature_K": float(env[-1, 1]), "tail_moisture": float(env[-1, 2])}
    radii = np.linspace(0, .02, 21)
    faces, centers, volumes = prefix["faces"], prefix["centers"], prefix["volumes"]
    args = (env, faces, centers, volumes, cfg["h"], cfg["hm"],
            cfg["atol"], cfg["rtol"], cfg["max_iterations"])
    T, C = prefix["final_T"].copy(), prefix["final_C"].copy()
    times = list(prefix["times"][::60])
    temperatures = list(prefix["temperature_K"][::60, ::5])
    moistures = list(prefix["moisture"][::60, ::5])
    max_C, max_r = [float("nan")]*len(times), [float("nan")]*len(times)
    max_C[-1], max_r[-1], _, _ = global_maximum(T, C, 10800., env, faces, centers, cfg["h"], cfg["hm"])
    loss = float(prefix["logs"][:, 1].sum())
    initial_loss = loss
    balance = [float(np.dot(C, volumes)/volumes.sum()+loss-2.55)]
    t, started, last_print = 10800., time.perf_counter(), time.perf_counter()
    while t < env[-1, 0]:
        stop = t+60.
        newT, newC, diag = advance_interval(T, C, t, stop, 1., *args)
        maximum, location, _, _ = global_maximum(newT, newC, stop, env, faces, centers, cfg["h"], cfg["hm"])
        crossed = False
        if maximum < .15:
            event = refine_endpoint(T, C, t, stop, env, faces, centers, volumes, cfg)
            newT, newC, diag, stop = event["T"], event["C"], event["diag"], event["time"]
            maximum, location, crossed = event["M"], event["r"], event["crossed"]
        T, C, t = newT, newC, stop
        loss += float(diag[1])
        st, sc = sample_fields(T, C, t, env, faces, centers, radii, cfg["h"], cfg["hm"])
        times.append(t); temperatures.append(st); moistures.append(sc)
        max_C.append(maximum); max_r.append(location)
        balance.append(float(np.dot(C, volumes)/volumes.sum()+loss-2.55))
        if crossed:
            break
        if time.perf_counter()-last_print > 30:
            print(f"Boundary control: t={t/3600:.3f}h max_C={maximum:.6f}", flush=True)
            last_print = time.perf_counter()
    else:
        raise RuntimeError("Boundary control reached its computational bound without drying")
    cfg.update(finish_time_s=t, finish_time_h=t/3600, bracket=event["bracket"],
               elapsed_seconds=time.perf_counter()-started,
               max_water_balance_residual=float(np.max(np.abs(balance))),
               q2_cumulative_loss=initial_loss, final_cumulative_loss=loss,
               source_q2_sha256=hashlib.sha256((ROOT/"q2/solution.npz").read_bytes()).hexdigest(),
               source_attachment_sha256=hashlib.sha256((ROOT/"data/attachment1.xlsx").read_bytes()).hexdigest())
    assert event["bracket"]["g_minus"] >= 0 and event["bracket"]["g_plus"] < 0
    assert np.max(np.abs(balance)) < 1e-8
    result = dict(times=np.asarray(times), radii=radii, temperature_K=np.asarray(temperatures),
                  moisture=np.asarray(moistures), max_C=np.asarray(max_C), max_r=np.asarray(max_r),
                  faces=faces, centers=centers, volumes=volumes, final_T=T, final_C=C,
                  environment=env, continuation_water_balance=np.asarray(balance), metadata=cfg)
    out = ROOT/"validation/q3_boundary_lastvalue.npz"
    save_solution(result, out)
    baseline_path = ROOT/"q3/runs/q3_N960_dt1_production_evdt0.05_tol0.01.npz"
    baseline = load_solution(baseline_path)
    return summarize(result, baseline, baseline_path, out)


def summarize(result, baseline, baseline_path, out):
    cfg = result["metadata"]
    t = cfg["finish_time_s"]
    comparison = difference(baseline, result, "boundary_assumption")
    # This comparison has no discretization pass/fail target: it changes an input.
    for key in ("targets", "C_passed", "C_also_below_1e_5", "endpoint_passed", "passed"):
        comparison.pop(key, None)
    comparison["interpretation"] = "Matched discretization with different post-4h boundary inputs; not numerical error"
    before = result["times"] <= 14400
    n_before = int(np.count_nonzero(before))
    unchanged = {key: float(np.max(np.abs(result[key][before]-baseline[key][:n_before, ::5])))
                 for key in ("temperature_K", "moisture")}
    assert max(unchanged.values()) < 1e-12
    report = {"scope": "Boundary assumption sensitivity at matched N=960 and dt=1s, not formal precision",
              "baseline_path": str(baseline_path.relative_to(ROOT)),
              "control_path": str(out.relative_to(ROOT)), "common_step_s": 1.,
              "unchanged_before_s": 14400., "before_4h_maximum_difference": unchanged,
              "baseline_tail": {"temperature_K": 323.15, "moisture": .05},
              "control_tail": {"temperature_K": cfg["tail_temperature_K"], "moisture": cfg["tail_moisture"]},
              "comparison": comparison, "control_metadata": cfg,
              "signed_finish_change_s": t-baseline["metadata"]["finish_time_s"],
              "relative_finish_change_percent": 100*(t/baseline["metadata"]["finish_time_s"]-1),
              "interpretation": "One alternative boundary continuation; not calibration, not experimental validation, and not a reason to change the predefined formal boundary"}
    (ROOT/"validation/q3_boundary_sensitivity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
