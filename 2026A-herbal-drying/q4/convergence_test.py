"""Independent Q4 refinements from t=0, with moving-mask-aware comparisons.

Every ordinary run solves Appendix 4 on its own original grid and initial
state. Physical coordinates, reference coordinates, and the actual surface
are all compared. Exterior NaNs cannot hide an interior comparison failure.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import time
import numpy as np
from common.solver import load_solution
from validation.q4_checks import LIMITS, matching_indices, check_solution, expected_mask

ROOT = bootstrap.ROOT
FIXED_ENDPOINT_LIMIT = 1.0


def job(cells=960, dt=1., startup=768., event_dt=.05, event_tol=.01,
        shrinking=True, tight=False):
    label = (f"q4_N{cells}_dt{dt:g}_startup{startup:g}_"
             f"{'shrinking' if shrinking else 'fixed'}_evdt{event_dt:g}_tol{event_tol:g}")
    if tight:
        label += "_tight"
    return {"cells": cells, "dt": dt, "startup": startup,
            "event_dt": event_dt, "event_tol": event_tol,
            "shrinking": shrinking, "tight": tight, "label": label}


def run_case(spec, force=False):
    from q4.solve_q4 import solve_q4
    started = time.perf_counter()
    path = ROOT/"q4/runs"/(spec["label"]+".npz")
    extra = {"atol": 1e-12, "rtol": 1e-13} if spec["tight"] else {}
    print(f"START {spec['label']}", flush=True)
    solve_q4(cells=spec["cells"], dt_max=spec["dt"], dt_scale=1/spec["startup"],
        event_dt=spec["event_dt"], event_tol=spec["event_tol"],
        shrinking=spec["shrinking"], output_path=path, recompute=force,
        time_growth=600., progress=True, **extra)
    result = load_solution(path)
    checks = check_solution(result)
    meta = result["metadata"]
    print(f"DONE {spec['label']} finish={meta['finish_time_s']:.9f} s", flush=True)
    return {**spec, "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "requested_event_dt": spec["event_dt"], "event_dt": meta["event_dt"],
        "config": {key: meta[key] for key in ("cells", "dt_max", "dt_scale", "time_growth",
            "event_dt", "epsilon_t", "shrinking", "atol", "rtol")},
        "finish_time_s": meta["finish_time_s"], "checks": checks,
        "elapsed_this_call_s": time.perf_counter()-started}


def _field_difference(left, right, times, positions, mask=None, position_name="radius_cm"):
    assert left.shape == right.shape
    if mask is None:
        mask = np.ones(left.shape, dtype=bool)
    assert mask.shape == left.shape and np.any(mask)
    assert np.isfinite(left[mask]).all() and np.isfinite(right[mask]).all()
    delta = np.full(left.shape, -np.inf)
    delta[mask] = abs(left[mask]-right[mask])
    row, col = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return {"maximum": float(delta[row, col]), "time_s": float(times[row]),
            position_name: float(positions[col]), "values_compared": int(mask.sum())}


def difference(coarse, fine, kind="time", coarse_path=None, fine_path=None):
    """Compare actual common coordinates, exact masks, material fields and surface."""
    assert coarse["metadata"]["shrinking"] == fine["metadata"]["shrinking"]
    if kind.startswith("fixed_"):
        assert coarse["metadata"]["shrinking"] is False, "A fixed-control criterion cannot accept a shrinking run"
    assert coarse["metadata"]["model_question"] == fine["metadata"]["model_question"] == 4
    finishes = [float(s["metadata"]["finish_time_s"]) for s in (coarse, fine)]
    last = math.floor((min(finishes)+1e-9)/60)*60
    times = np.arange(0., last+1, 60.)
    radii = np.linspace(0., .02, 21)
    rows = [matching_indices(s["times"], times) for s in (coarse, fine)]
    cols = [matching_indices(s["radii"], radii, 1e-12) for s in (coarse, fine)]
    radius0, radius1 = coarse["radius_m"][rows[0]], fine["radius_m"][rows[1]]
    assert np.array_equal(radius0, radius1), "Matched physical times must have identical radius histories"
    masks = [s["valid_mask"][row][:, col] for s, row, col in zip((coarse, fine), rows, cols)]
    assert np.array_equal(masks[0], masks[1]), "Mask disagreement is a failure, not removable NaNs"
    assert np.array_equal(masks[0], expected_mask(radii, radius0))
    xi = np.linspace(0., 1., 101)
    refcols = [matching_indices(s["xi"], xi, 1e-12) for s in (coarse, fine)]
    config_keys = ("cells", "dt_max", "dt_scale", "time_growth", "event_dt", "epsilon_t", "shrinking", "atol", "rtol")
    report = {"kind": kind, "comparison_times": len(times), "comparison_radii": 21,
        "comparison_reference_positions": 101, "last_common_regular_time_s": last,
        "interpolation_used": False, "radius_histories_identical": True,
        "physical_masks_identical": True, "physical_inside_values": int(masks[0].sum()),
        "physical_outside_values": int((~masks[0]).sum()),
        "coarse_config": {k: coarse["metadata"][k] for k in config_keys},
        "fine_config": {k: fine["metadata"][k] for k in config_keys}}
    if coarse_path is not None and fine_path is not None:
        paths = [str(value).replace("\\", "/") for value in (coarse_path, fine_path)]
        report["case_paths"] = {"coarse": paths[0], "fine": paths[1]}
        report["coarse_path"], report["fine_path"] = paths
    if kind in ("time", "fixed_time"):
        report["time_step_ratio"] = coarse["metadata"]["dt_max"]/fine["metadata"]["dt_max"]
    if kind in ("startup", "fixed_startup"):
        report["startup_step_ratio"] = coarse["metadata"]["dt_scale"]/fine["metadata"]["dt_scale"]
    for physical_key, reference_key, surface_key, label in (
        ("moisture", "moisture_ref", "surface_moisture", "C"),
        ("temperature_K", "temperature_ref_K", "surface_temperature_K", "T"),
    ):
        a, b = coarse[physical_key][rows[0]][:, cols[0]], fine[physical_key][rows[1]][:, cols[1]]
        report[f"physical_{label}"] = _field_difference(a, b, times, radii*100, masks[0])
        a, b = coarse[reference_key][rows[0]][:, refcols[0]], fine[reference_key][rows[1]][:, refcols[1]]
        report[f"reference_{label}"] = _field_difference(a, b, times, xi, position_name="xi")
        a, b = coarse[surface_key][rows[0]], fine[surface_key][rows[1]]
        delta = abs(a-b)
        assert np.isfinite(delta).all()
        i = int(np.argmax(delta))
        report[f"surface_{label}"] = {"maximum": float(delta[i]), "time_s": float(times[i]),
                                       "values_compared": len(times)}
        candidates = {name: report[f"{name}_{label}"]["maximum"] for name in ("physical", "reference", "surface")}
        report[label] = {"maximum": max(candidates.values()), "maximum_by_view": candidates}
    max_delta = abs(coarse["max_C"][rows[0]]-fine["max_C"][rows[1]])
    assert np.isfinite(max_delta).all()
    i = int(np.argmax(max_delta))
    report["full_grid_maximum_C"] = {"maximum": float(max_delta[i]), "time_s": float(times[i])}
    report["C"]["maximum_by_view"]["full_grid_maximum"] = float(max_delta[i])
    report["C"]["maximum"] = max(report["C"]["maximum"], float(max_delta[i]))
    endpoint_difference = abs(finishes[0]-finishes[1])
    report["finish_time_s"] = {"coarse": finishes[0], "fine": finishes[1], "difference": endpoint_difference}
    report["at_own_endpoints_reference_C_difference"] = float(np.max(abs(
        coarse["moisture_ref"][-1, refcols[0]]-fine["moisture_ref"][-1, refcols[1]])))
    report["at_own_endpoints_surface_C_difference"] = float(abs(coarse["surface_moisture"][-1]-fine["surface_moisture"][-1]))
    endpoint_limit = LIMITS["event_s"] if kind in ("event_tolerance", "event_integration_step") else LIMITS["endpoint_s"]
    if kind.startswith("fixed_"):
        endpoint_limit = FIXED_ENDPOINT_LIMIT
    report["targets"] = {"full_output_C": LIMITS["field_C"], "endpoint_s": endpoint_limit}
    report["C_passed"] = report["C"]["maximum"] <= LIMITS["field_C"]
    report["endpoint_passed"] = endpoint_difference <= endpoint_limit
    report["passed"] = report["C_passed"] and report["endpoint_passed"]
    report["interpretation"] = "Observed numerical refinement differences, not an exact-solution bound or measured-field validation"
    return report


def matrix(kind, time_steps=(1., .5), dt=.5, startup=768., cells=960):
    if kind == "time":
        cases = [job(cells=cells, dt=step, startup=startup) for step in time_steps]
        pairs = [(kind, a["label"], b["label"]) for a, b in zip(cases[:-1], cases[1:])]
    elif kind == "space":
        cases = [job(cells=n, dt=dt, startup=startup) for n in (960, 1280, 1600)]
        pairs = [(kind, a["label"], b["label"]) for a, b in zip(cases[:-1], cases[1:])]
        pairs.append(("space_cross_two_levels", cases[0]["label"], cases[-1]["label"]))
    elif kind == "startup":
        cases = [job(cells=cells, dt=dt, startup=value) for value in (768., 1536.)]
        pairs = [(kind, cases[0]["label"], cases[1]["label"])]
    elif kind == "picard":
        cases = [job(cells=cells, dt=dt, startup=startup, tight=tight) for tight in (False, True)]
        pairs = [("picard_tolerance", cases[0]["label"], cases[1]["label"])]
    else:
        raise ValueError(f"Unsupported full-trajectory matrix: {kind}")
    return cases, pairs


def execute(kind="time", workers=2, force=False, time_steps=(1., .5), dt=.5,
            startup=768., cells=960, output=None):
    cases, pairs = matrix(kind, time_steps, dt, startup, cells)
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
        paths = [results[label]["path"] for label in (left, right)]
        try:
            comparison = difference(load_solution(ROOT/paths[0]), load_solution(ROOT/paths[1]), category, *paths)
            comparisons.append({"coarse": left, "fine": right, **comparison})
        except (AssertionError, ValueError) as exc:
            comparisons.append({"kind": category, "coarse": left, "fine": right,
                "case_paths": {"coarse": paths[0], "fine": paths[1]}, "passed": False,
                "comparison_failed": {"type": type(exc).__name__, "message": str(exc)}})
    report = {"matrix": kind, "predeclared_limits": LIMITS, "runs": list(results.values()),
        "failed_runs": failures, "comparisons": comparisons,
        "all_requested_comparisons_passed": not failures and all(item["passed"] for item in comparisons),
        "source_policy": "Every grid is solved from t=0 and the original Appendix 4 initial state; no seed projection",
        "claim_boundary": "Separate observed numerical differences; not a combined exact-solution error bound"}
    path = Path(output or ROOT/"validation"/f"q4_convergence_{kind}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "comparisons": comparisons, "failed_runs": failures},
                     ensure_ascii=False, indent=2), flush=True)
    return report


def local_event_checks(base_path, kind="event", output=None):
    """Reintegrate just the event from one stored full material-state anchor."""
    from common.solver import save_solution
    from q4.end_time_detector import refine_endpoint
    from q4.kernel import sample_fields, surface_fields
    from q4.shrinking_radius import radius_at
    base_path = (ROOT/Path(str(base_path).replace("\\", "/"))).resolve()
    base = load_solution(base_path)
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    anchor = float(base["event_anchor_time_s"])
    assert base["times"][-2] == anchor
    stop = (math.floor(anchor/60)+1)*60.
    effective_dt = float(base["metadata"]["event_dt"])
    settings = [(effective_dt, .01), (effective_dt, .001)] if kind == "event" else [(effective_dt, .001), (effective_dt/2, .001)]
    records, results = [], []
    for event_dt, tolerance in settings:
        started = time.perf_counter()
        config = {**base["metadata"], "event_dt": event_dt, "epsilon_t": tolerance}
        event = refine_endpoint(base["event_anchor_T"], base["event_anchor_C"], anchor, stop,
            base["environment"], base["faces"], base["centers"], base["volumes"],
            config, base["radius_data"], base["radius_slopes"])
        if not event["crossed"]:
            raise RuntimeError("Saved event interval does not bracket the refined trajectory")
        result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base.items()}
        result["metadata"] = config
        end = float(event["time"])
        R = float(radius_at(end, base["radius_data"], base["radius_slopes"], bool(config["shrinking"])))
        config.update(finish_time_s=end, finish_time_h=end/3600, last_time_s=end, bracket=event["bracket"],
            final_radius_m=R, validation_parent_path=base_path.relative_to(ROOT).as_posix(),
            validation_parent_sha256=base_hash,
            validation_parent_cache_fingerprint=config.pop("cache_fingerprint", None),
            validation_scope="same saved full-state anchor; only the terminal event subproblem",
            validation_prefix_reused=True, elapsed_seconds=time.perf_counter()-started)
        args = (config["h"], config["hm"], base["radius_data"], base["radius_slopes"], bool(config["shrinking"]))
        physicalT, physicalC, mask = sample_fields(event["T"], event["C"], end, base["environment"],
            base["faces"], base["centers"], base["radii"], *args, False)
        referenceT, referenceC, _ = sample_fields(event["T"], event["C"], end, base["environment"],
            base["faces"], base["centers"], base["reference_radii"], *args, True)
        surfaceT, surfaceC = surface_fields(event["T"], event["C"], end, base["environment"],
            base["faces"], base["centers"], *args)
        for key, value in (("times", end), ("temperature_K", physicalT), ("moisture", physicalC),
            ("valid_mask", mask), ("temperature_ref_K", referenceT), ("moisture_ref", referenceC),
            ("surface_temperature_K", surfaceT), ("surface_moisture", surfaceC), ("radius_m", R),
            ("reference_physical_radii", R*base["xi"]), ("max_C", event["M"]), ("max_r", event["r"]),
            ("logs", event["diag"]), ("mean_C", event["diag"][0])):
            result[key][-1] = value
        result["final_T"], result["final_C"] = event["T"], event["C"]
        result["final_radius_m"] = np.asarray(R)
        result["final_physical_faces"], result["final_physical_centers"] = R/.02*base["faces"], R/.02*base["centers"]
        anchor_loss = float(base["event_anchor_loss_cumulative"])
        result["loss_cumulative"][-1] = anchor_loss+event["diag"][1]
        result["water_balance"][-1] = result["mean_C"][-1]+result["loss_cumulative"][-1]-2.55
        result["event_trace"] = event["trace"]
        result["event_lower_T"], result["event_lower_C"] = event["lower_T"], event["lower_C"]
        result["event_lower_time_s"] = np.asarray(event["bracket"]["t_minus"])
        result["event_lower_loss_cumulative"] = np.asarray(anchor_loss+event["lower_diag"][1])
        result["event_lower_radius_m"] = np.asarray(radius_at(event["bracket"]["t_minus"],
            base["radius_data"], base["radius_slopes"], bool(config["shrinking"])))
        result["event_upper_T"], result["event_upper_C"] = event["T"].copy(), event["C"].copy()
        result["event_upper_time_s"] = np.asarray(end)
        for key, column in (("steps", 2), ("iterations", 3), ("rejections", 4), ("damped_iterations", 8)):
            config[key] = int(np.sum(result["logs"][:, column]))
        config["max_water_balance_residual"] = float(np.max(abs(result["water_balance"])))
        config["max_scaled_residual"] = float(np.max(result["logs"][:, 7]))
        config["event_work_steps"], config["event_work_iterations"] = event["work_steps"], event["work_iterations"]
        mode = "shrinking" if config["shrinking"] else "fixed"
        label = (f"q4_local_{kind}_N{config['cells']}_dt{config['dt_max']:g}_startup{1/config['dt_scale']:g}_"
                 f"{mode}_evdt{event_dt:g}_tol{tolerance:g}")
        path = ROOT/"q4/runs"/(label+".npz")
        save_solution(result, path)
        checks = check_solution(load_solution(path))
        records.append({"label": label, "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "base_path": base_path.relative_to(ROOT).as_posix(), "base_sha256": base_hash,
            "anchor_time_s": anchor, "anchor_loss_cumulative": anchor_loss,
            "local_work_steps_including_discarded_candidates": event["work_steps"],
            "local_work_iterations_including_discarded_candidates": event["work_iterations"],
            "accepted_local_interval_loss": float(event["diag"][1]),
            "accepted_local_steps": int(event["diag"][2]),
            "event_dt": event_dt, "event_tol": tolerance, "checks": checks})
        results.append(result)
    category = "event_tolerance" if kind == "event" else "event_integration_step"
    comparison = {"coarse": records[0]["label"], "fine": records[1]["label"],
        **difference(results[0], results[1], category, records[0]["path"], records[1]["path"])}
    report = {"matrix": kind, "predeclared_limits": LIMITS, "runs": records,
        "failed_runs": {}, "comparisons": [comparison],
        "all_requested_comparisons_passed": comparison["passed"],
        "scope": "Identical full material anchor and unchanged prefix; only event tolerance or step changed",
        "claim_boundary": "Event subproblem verification, not whole-trajectory convergence"}
    path = Path(output or ROOT/"validation"/f"q4_convergence_{kind}.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("time", "space", "startup", "picard", "event", "event_step"), default="time")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--time-steps", type=float, nargs="+", default=[1., .5])
    parser.add_argument("--dt", type=float, default=.5)
    parser.add_argument("--startup", type=float, default=768.)
    parser.add_argument("--cells", type=int, default=960)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--event-base", type=Path)
    args = parser.parse_args()
    if args.matrix in ("event", "event_step"):
        if args.event_base is None:
            parser.error("Event studies require a completed --event-base with a saved native anchor")
        report = local_event_checks(args.event_base, args.matrix, args.output)
    else:
        report = execute(args.matrix, args.workers, args.force, tuple(args.time_steps),
            args.dt, args.startup, args.cells, args.output)
    if args.require_pass and not report["all_requested_comparisons_passed"]:
        raise SystemExit(1)
