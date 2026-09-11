"""Independent Q3 refinements on every 60-second/0.1-cm output location.

Time runs keep the exact same Q2 endpoint. Space runs use their own existing
Q2 endpoint at the original grid, with a matched Q2 time-discretization scale.
No initial-state projection, time interpolation, clipping or extrapolated
"best" solution is used. Coarse failed comparisons remain in the report.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import time
import hashlib
import numpy as np
from common.solver import load_solution
from validation.q3_checks import LIMITS, matching_indices, check_solution

ROOT = bootstrap.ROOT


def seed_path(cells):
    """Existing equally time-refined Q2 runs; never regenerate or project them."""
    path = ROOT/"validation/runs"/f"p10cap4_q2_N{cells}_dt0.0009765625_growth600.npz"
    if not path.is_file():
        raise FileNotFoundError(f"A complete original-grid Q2 seed is required: {path}")
    return path


def job(cells=960, dt=1., event_dt=.05, event_tol=.01, source="production", tight=False):
    label = f"q3_N{cells}_dt{dt:g}_{source}_evdt{event_dt:g}_tol{event_tol:g}"
    if tight:
        label += "_tight"
    return {"cells": cells, "dt": dt, "event_dt": event_dt, "event_tol": event_tol,
            "source": source, "tight": tight, "label": label}


def run_case(spec, force=False):
    from q3.solve_q3 import solve_q3
    source = None if spec["source"] == "production" else seed_path(spec["cells"])
    if source is None and spec["cells"] != 960:
        raise ValueError("Production Q2 source is N=960; use original-grid seeds for space refinement")
    source_for_check = ROOT/"q2/solution.npz" if source is None else source
    seed = load_solution(source_for_check)
    assert seed["metadata"]["question"] == 2
    assert seed["metadata"]["cells"] == spec["cells"]
    path = ROOT/"q3/runs"/(spec["label"]+".npz")
    print(f"START {spec['label']}", flush=True)
    started = time.perf_counter()
    extra = {}
    if spec["tight"]:
        extra = {"atol": seed["metadata"]["atol"]*.01, "rtol": seed["metadata"]["rtol"]*.01}
    result = solve_q3(cells=spec["cells"], dt_max=spec["dt"], event_dt=spec["event_dt"],
                      event_tol=spec["event_tol"], source_path=source, output_path=path,
                      recompute=force, progress=True, **extra)
    # The public solver returns a result dictionary; saved arrays are also
    # reloaded to ensure the persisted object is the one that is compared.
    persisted = load_solution(path)
    if persisted["metadata"].get("status") != "complete":
        raise RuntimeError(f"{spec['label']} did not reach a strict threshold event")
    check = check_solution(persisted, seed)
    finish = persisted["metadata"]["finish_time_s"]
    print(f"DONE {spec['label']} finish={finish:.9f}s elapsed={time.perf_counter()-started:.2f}s", flush=True)
    return {**spec, "requested_event_dt": spec["event_dt"],
            "event_dt": float(persisted["metadata"]["event_dt"]),
            "path": str(path), "source_path": str(source_for_check),
            "finish_time_s": finish, "checks": check,
            "elapsed_this_call_s": time.perf_counter()-started}


def difference(coarse, fine, kind="time"):
    finish0 = float(coarse["metadata"]["finish_time_s"])
    finish1 = float(fine["metadata"]["finish_time_s"])
    last = math.floor((min(finish0, finish1)+1e-9)/60)*60
    times = np.arange(0., last+1, 60.)
    radii = np.linspace(0., .02, 21)
    rows0 = matching_indices(coarse["times"], times)
    rows1 = matching_indices(fine["times"], times)
    cols0 = matching_indices(coarse["radii"], radii, 1e-12)
    cols1 = matching_indices(fine["radii"], radii, 1e-12)
    report = {"kind": kind, "comparison_times": len(times), "comparison_radii": 21,
              "last_common_regular_time_s": last, "interpolation_used": False}
    report["coarse_config"] = {k: coarse["metadata"][k] for k in ("cells", "dt_max", "event_dt", "epsilon_t")}
    report["fine_config"] = {k: fine["metadata"][k] for k in ("cells", "dt_max", "event_dt", "epsilon_t")}
    if kind == "time":
        report["time_step_ratio"] = coarse["metadata"]["dt_max"]/fine["metadata"]["dt_max"]
        report["event_step_policy"] = "Terminal event step is min(requested event_dt, main dt); its effect is checked independently from a common anchor"
    for key, label in (("moisture", "C"), ("temperature_K", "T")):
        delta = abs(coarse[key][rows0][:, cols0]-fine[key][rows1][:, cols1])
        i, j = np.unravel_index(int(np.argmax(delta)), delta.shape)
        report[label] = {"maximum": float(delta[i, j]), "time_s": float(times[i]),
                         "radius_cm": float(radii[j]*100)}
        late = times >= 10800
        report[label]["post_3h_maximum"] = float(delta[late].max())
    endpoint_difference = abs(finish0-finish1)
    report["finish_time_s"] = {"coarse": finish0, "fine": finish1, "difference": endpoint_difference}
    # These endpoint profiles belong to different times; record them separately
    # instead of presenting their difference as a same-time discretization error.
    report["at_own_endpoints_C_profile_difference"] = float(np.max(abs(
        coarse["moisture"][-1, cols0]-fine["moisture"][-1, cols1])))
    event_limit = LIMITS["event_s"] if kind in ("event_tolerance", "event_integration_step") else LIMITS["endpoint_s"]
    report["targets"] = {"full_output_C": LIMITS["field_C"], "endpoint_s": event_limit}
    report["C_passed"] = report["C"]["maximum"] <= LIMITS["field_C"]
    report["C_also_below_1e_5"] = report["C"]["maximum"] <= 1e-5
    report["endpoint_passed"] = endpoint_difference <= event_limit
    report["passed"] = report["C_passed"] and report["endpoint_passed"]
    report["interpretation"] = "Difference between numerical resolutions; not error against an exact solution or experimental measurement"
    return report


def local_event_checks(base_path, kind="event", output=None):
    """Isolate event-location error using the SAME saved undried full state.

    Only the final partial-minute trajectory is repeated. This avoids rerunning
    the identical 57-hour prefix and does not substitute interpolated events.
    """
    from common.data_loader import load_environment
    from common.continuation import sample_fields
    from common.solver import save_solution
    from q3.end_time_detector import refine_endpoint
    base_path = Path(base_path).resolve()
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    base = load_solution(base_path)
    source = load_solution(base["metadata"]["source_path"])
    env = load_environment(ROOT/"data/attachment1.xlsx")
    anchor = float(base["event_anchor_time_s"])
    assert abs(base["times"][-2]-anchor) <= 1e-9
    endpoint_stop = (math.floor(anchor/60)+1)*60.
    base_event_dt = float(base["metadata"]["event_dt"])
    settings = [(base_event_dt, .01), (base_event_dt, .001)] if kind == "event" else [(base_event_dt, .001), (base_event_dt/2, .001)]
    results, records = [], []
    for event_dt, tolerance in settings:
        started = time.perf_counter()
        cfg = {**base["metadata"], "event_dt": event_dt, "epsilon_t": tolerance}
        event = refine_endpoint(base["event_anchor_T"], base["event_anchor_C"], anchor, endpoint_stop,
                                env, base["faces"], base["centers"], base["volumes"], cfg)
        if not event["crossed"]:
            raise RuntimeError("The common event anchor interval no longer brackets a crossing; extend the independent event study explicitly")
        result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in base.items()}
        result["metadata"] = cfg
        cfg.update({"finish_time_s": event["time"], "finish_time_h": event["time"]/3600,
                    "last_time_s": event["time"], "bracket": event["bracket"],
                    "validation_parent_path": str(base_path),
                    "validation_parent_sha256": base_hash,
                    "validation_parent_cache_fingerprint": cfg.pop("cache_fingerprint", None),
                    "validation_scope": "same saved final-interval anchor; event subproblem only",
                    "elapsed_seconds": time.perf_counter()-started})
        result["times"][-1] = event["time"]
        sampleT, sampleC = sample_fields(event["T"], event["C"], event["time"], env,
            base["faces"], base["centers"], base["radii"], cfg["h"], cfg["hm"])
        result["temperature_K"][-1], result["moisture"][-1] = sampleT, sampleC
        result["final_T"], result["final_C"] = event["T"], event["C"]
        result["max_C"][-1], result["max_r"][-1] = event["M"], event["r"]
        result["logs"][-1] = event["diag"]
        result["mean_C"][-1] = event["diag"][0]
        result["loss_cumulative"][-1] = float(base["event_anchor_loss_cumulative"])+event["diag"][1]
        result["water_balance"][-1] = result["mean_C"][-1]+result["loss_cumulative"][-1]-2.55
        result["event_trace"] = event["trace"]
        result["event_lower_T"], result["event_lower_C"] = event["lower_T"], event["lower_C"]
        result["event_lower_loss_cumulative"] = np.array(float(base["event_anchor_loss_cumulative"])+event["lower_diag"][1])
        cfg["max_water_balance_residual"] = float(np.max(abs(result["water_balance"])))
        post_logs = result["logs"][result["times"] > 10800]
        for key, column in (("steps", 2), ("iterations", 3), ("rejections", 4), ("damped_iterations", 8)):
            cfg[key] = int(np.sum(post_logs[:, column]))
        cfg["max_scaled_residual"] = float(np.max(result["logs"][:, 7]))
        cfg["event_work_steps"], cfg["event_work_iterations"] = event["work_steps"], event["work_iterations"]
        cfg["validation_prefix_reused"] = True
        label = f"local_event_N{cfg['cells']}_dt{cfg['dt_max']:g}_evdt{event_dt:g}_tol{tolerance:g}"
        path = ROOT/"q3/runs"/(label+".npz")
        save_solution(result, path)
        checks = check_solution(load_solution(path), source)
        records.append({"label": label, "path": str(path), "base_path": str(base_path),
                        "base_sha256": base_hash, "anchor_time_s": anchor,
                        "anchor_loss_cumulative": float(base["event_anchor_loss_cumulative"]),
                        "local_work_steps_including_rejected_candidates": event["work_steps"],
                        "local_work_iterations_including_rejected_candidates": event["work_iterations"],
                        "accepted_local_interval_loss": float(event["diag"][1]),
                        "accepted_local_steps": int(event["diag"][2]),
                        "event_dt": event_dt, "event_tol": tolerance, "checks": checks,
                        "elapsed_this_call_s": time.perf_counter()-started})
        results.append(result)
    category = "event_tolerance" if kind == "event" else "event_integration_step"
    comparison = {"coarse": records[0]["label"], "fine": records[1]["label"],
                  **difference(results[0], results[1], category)}
    report = {"matrix": kind, "predeclared_limits": LIMITS, "runs": records, "failed_runs": {},
              "comparisons": [comparison], "all_requested_comparisons_passed": comparison["passed"],
              "scope": "Identical full-state anchor and unchanged full prefix; only event tolerance/step is changed",
              "claim_boundary": "Event subproblem convergence, not whole-trajectory discretization accuracy"}
    path = Path(output or ROOT/"validation"/f"q3_convergence_{kind}.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def matrix(kind, time_steps=(1., .5, .25), space_step=.25, event_step=.25):
    cases, pairs = {}, []

    def add(spec):
        cases[spec["label"]] = spec
        return spec["label"]

    if kind in ("all", "time"):
        ids = [add(job(dt=dt)) for dt in time_steps]
        pairs.extend(("time", a, b) for a, b in zip(ids[:-1], ids[1:]))
    if kind in ("all", "space"):
        ids = [add(job(cells=n, dt=space_step, source="matched_q2")) for n in (960, 1280, 1600)]
        pairs.extend(("space", a, b) for a, b in zip(ids[:-1], ids[1:]))
        pairs.append(("space_cross_two_levels", ids[0], ids[-1]))
        a = add(job(dt=space_step))
        pairs.append(("q2_seed_sensitivity", a, ids[0]))
    if kind in ("all", "event"):
        ids = [add(job(dt=event_step, event_tol=tol)) for tol in (.01, .001)]
        pairs.append(("event_tolerance", *ids))
    if kind == "event_step":
        effective_event_step = min(.05, event_step)
        ids = [add(job(dt=event_step, event_dt=dt, event_tol=.001))
               for dt in (effective_event_step, effective_event_step/2)]
        pairs.append(("event_integration_step", *ids))
    if kind == "picard":
        ids = [add(job(dt=event_step, tight=tight)) for tight in (False, True)]
        pairs.append(("picard_tolerance", *ids))
    return list(cases.values()), pairs


def execute(kind="all", workers=2, force=False, time_steps=(1., .5, .25),
            space_step=.25, event_step=.25, output=None):
    cases, pairs = matrix(kind, time_steps, space_step, event_step)
    results, failures = {}, {}
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 3))) as pool:
        futures = {pool.submit(run_case, spec, force): spec for spec in cases}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results[spec["label"]] = future.result()
            except Exception as exc:
                failures[spec["label"]] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"FAILED {spec['label']}: {type(exc).__name__}: {exc}", flush=True)
    comparisons = []
    for category, left, right in pairs:
        if left not in results or right not in results:
            comparisons.append({"kind": category, "coarse": left, "fine": right,
                                "passed": False, "unavailable_run": True})
            continue
        diff = difference(load_solution(results[left]["path"]), load_solution(results[right]["path"]), category)
        comparisons.append({"coarse": left, "fine": right, **diff})
    report = {"matrix": kind, "predeclared_limits": LIMITS, "runs": list(results.values()),
              "failed_runs": failures, "comparisons": comparisons,
              "all_requested_comparisons_passed": not failures and all(x["passed"] for x in comparisons),
              "source_policy": "Original per-grid appendix-3 Q2 states; no state projection; matched Q2 temporal settings for spatial matrix",
              "claim_boundary": "Numerical stability and artifact consistency only; supplied input data are not internal-field validation measurements"}
    path = Path(output or ROOT/"validation"/f"q3_convergence_{kind}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "comparisons": comparisons, "failed_runs": failures}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=("all", "time", "space", "event", "event_step", "picard"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--time-steps", type=float, nargs="+", default=[1., .5, .25])
    parser.add_argument("--space-step", type=float, default=.25)
    parser.add_argument("--event-step", type=float, default=.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--event-base", type=Path, help="Reuse only the saved event anchor, preserving the exact full prefix")
    args = parser.parse_args()
    if args.event_base is not None:
        if args.matrix not in ("event", "event_step"):
            parser.error("--event-base is supported only for event/event_step studies")
        report = local_event_checks(args.event_base, args.matrix, args.output)
    else:
        report = execute(args.matrix, args.workers, args.force, tuple(args.time_steps),
                         args.space_step, args.event_step, args.output)
    if args.require_pass and not report["all_requested_comparisons_passed"]:
        raise SystemExit(1)
