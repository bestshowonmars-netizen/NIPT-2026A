"""Solve Appendix 4 from its initial state on a fixed material reference mesh."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import json
import time
import numpy as np
from common.data_loader import load_environment
from common.solver import load_solution, save_solution
from common.finite_volume import geometry
from q4.config import make_config
from q4.kernel import advance_interval, global_maximum, sample_fields, surface_fields
from q4.shrinking_radius import load_radius_data, radius_at
from q4.end_time_detector import refine_endpoint

ROOT = bootstrap.ROOT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_label(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _fingerprint(config, cells, environment_path, radius_path):
    paths = [environment_path, radius_path]
    paths.extend(ROOT/name for name in (
        "common/solver.py", "common/finite_volume.py", "common/material_properties.py",
        "common/boundary_conditions.py", "common/interpolation.py", "common/data_loader.py",
        "q4/config.py", "q4/kernel.py", "q4/shrinking_radius.py",
        "q4/end_time_detector.py", "q4/solve_q4.py"))
    hashes = {_source_label(path): file_sha256(path) for path in paths}
    payload = json.dumps({"config": config, "cells": cells, "files": hashes},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), hashes


def _samples(T, C, t, environment, faces, centers, radii, reference_radii,
             radius_data, radius_slopes, config):
    args = (config["h"], config["hm"], radius_data, radius_slopes, config["shrinking"])
    physical_T, physical_C, mask = sample_fields(T, C, t, environment, faces, centers,
                                                radii, *args, False)
    reference_T, reference_C, reference_mask = sample_fields(T, C, t, environment,
        faces, centers, reference_radii, *args, True)
    surface_T, surface_C = surface_fields(T, C, t, environment, faces, centers, *args)
    radius = float(radius_at(t, radius_data, radius_slopes, config["shrinking"], config["radius"]))
    if not 0.0 < radius <= config["radius"]+1e-14:
        raise RuntimeError("Q4 radius is outside the supplied shrinking domain")
    if not (np.array_equal(mask, (radii >= 0.0) & (radii <= radius))
            and np.all(reference_mask)):
        raise RuntimeError("Physical-domain mask or reference sampling is inconsistent")
    if not (np.isfinite(physical_T[mask]).all() and np.isfinite(physical_C[mask]).all()
            and np.isnan(physical_T[~mask]).all() and np.isnan(physical_C[~mask]).all()
            and np.isfinite(reference_T).all() and np.isfinite(reference_C).all()
            and np.isfinite(surface_T) and np.isfinite(surface_C)):
        raise RuntimeError("Q4 samples must be finite inside the domain and NaN outside")
    return (physical_T, physical_C, mask, reference_T, reference_C,
            float(surface_T), float(surface_C), radius)


def solve_q4(cells=960, dt_max=1.0, dt_scale=1.0/768.0, event_dt=0.05,
             event_tol=0.01, shrinking=True, output_path=None, recompute=False,
             progress=True, end_limit=72.0*3600.0, hard_end_limit=30.0*24.0*3600.0,
             atol=None, rtol=None, sample_count=101, time_growth=600.0,
             environment_path=None, radius_path=None):
    """Run independently from t=0; a hard limit saves an explicitly incomplete run.

    Native faces/centers/volumes use reference x in metres, x in [0,R0].
    Physical output positions are fixed; exterior values are NaN with a mask.
    Reference samples remain complete as their physical positions move with R(t).
    """
    started = time.perf_counter()
    config = make_config(dt_max, dt_scale, event_dt, event_tol, shrinking, end_limit,
                         hard_end_limit, atol, rtol, sample_count, time_growth)
    if int(cells) != cells or cells < 2:
        raise ValueError("At least two integer native cells are required for center reconstruction")
    cells = int(cells)
    environment_path = Path(environment_path or ROOT/"data/attachment1.xlsx").resolve()
    radius_path = Path(radius_path or ROOT/"data/attachment2.xlsx").resolve()
    output_path = Path(output_path) if output_path is not None else ROOT/(
        "q4/solution.npz" if config["shrinking"] else "q4/fixed_control.npz")
    fingerprint, hashes = _fingerprint(config, cells, environment_path, radius_path)
    if output_path.exists() and not recompute:
        cached = load_solution(output_path)
        if cached["metadata"].get("cache_fingerprint") == fingerprint:
            if progress:
                print(f'Q4 cache: {cached["metadata"].get("status")}, {output_path}', flush=True)
            return cached
    environment = load_environment(environment_path)
    radius_data, radius_slopes = load_radius_data(radius_path)
    if not np.isclose(radius_data[0, 1], config["radius"], rtol=0, atol=1e-14):
        raise ValueError("Attachment 2 initial radius must match the Appendix 4 initial domain")
    faces, centers, volumes = geometry(cells, config["radius"])
    initial_T = np.full(cells, config["initial_T"])
    initial_C = np.full(cells, config["initial_C"])
    T, C = initial_T.copy(), initial_C.copy()
    radii = np.linspace(0.0, config["radius"], config["sample_count"])
    reference_radii = radii.copy()
    xi = reference_radii/config["radius"]
    t = config["start_time"]
    spacing = config["output_interval"]
    times, temperatures, moistures, masks = [], [], [], []
    reference_temperatures, reference_moistures = [], []
    surface_temperatures, surface_moistures, radius_history = [], [], []

    def append_samples(temperature, moisture, when):
        sampled = _samples(temperature, moisture, when, environment, faces, centers,
                           radii, reference_radii, radius_data, radius_slopes, config)
        pt, pc, mask, rt, rc, st, sc, radius = sampled
        if times and not when > times[-1]:
            raise RuntimeError("Q4 output times must increase without duplicate endpoint rows")
        times.append(float(when))
        temperatures.append(pt); moistures.append(pc); masks.append(mask)
        reference_temperatures.append(rt); reference_moistures.append(rc)
        surface_temperatures.append(st); surface_moistures.append(sc)
        radius_history.append(radius)

    append_samples(T, C, t)
    M, rmax, _, _ = global_maximum(T, C, t, environment, faces, centers,
        config["h"], config["hm"], radius_data, radius_slopes, config["shrinking"])
    if not np.isfinite(M) or M < config["threshold"]:
        raise ValueError("The supplied initial state must be finite and not already below threshold")
    maxima, maximum_radii = [float(M)], [float(rmax)]
    means, losses = [config["initial_C"]], [0.0]
    initial_log = np.zeros(9)
    initial_log[0] = config["initial_C"]
    logs = [initial_log]
    horizon = config["initial_end_limit"]
    extended_limits = [horizon]
    complete, bracket = False, None
    event_trace = []
    event_anchor_T, event_anchor_C = np.empty(0), np.empty(0)
    event_lower_T, event_lower_C = np.empty(0), np.empty(0)
    event_anchor_time = event_anchor_loss = event_lower_loss = float("nan")
    event_work_steps = event_work_iterations = discarded_coarse_steps = 0
    last_report = started
    if progress:
        print(f"Q4 start: N={cells}, shrinking={config['shrinking']}, "
              f"dt<={config['dt_max']:g} s, startup={config['dt_scale']:g} s", flush=True)
    while t < config["hard_end_limit"]-1e-10:
        target = min((np.floor(t/spacing)+1.0)*spacing, config["hard_end_limit"])
        while target > horizon and horizon < config["hard_end_limit"]:
            horizon = min(config["hard_end_limit"], horizon*config["horizon_growth"])
            extended_limits.append(horizon)
            if progress:
                print(f"Q4 remains undried; extend search to {horizon/3600:g} h", flush=True)
        newT, newC, diag = advance_interval(T, C, t, target, config["dt_max"], environment,
            faces, centers, volumes, config["h"], config["hm"], config["atol"], config["rtol"],
            config["max_iterations"], radius_data, radius_slopes, config["shrinking"],
            config["dt_scale"], config["time_growth"])
        M, rmax, _, _ = global_maximum(newT, newC, target, environment, faces, centers,
            config["h"], config["hm"], radius_data, radius_slopes, config["shrinking"])
        if not np.isfinite(M) or not np.isfinite(diag).all():
            raise RuntimeError("Q4 produced a nonfinite maximum or diagnostic")
        if M < config["threshold"]:
            event = refine_endpoint(T, C, t, target, environment, faces, centers, volumes,
                                    config, radius_data, radius_slopes)
            event_trace.extend(event["trace"].tolist())
            event_work_steps += event["work_steps"]
            event_work_iterations += event["work_iterations"]
            discarded_coarse_steps += int(diag[2])
            newT, newC, diag = event["T"], event["C"], event["diag"]
            M, rmax, target = event["M"], event["r"], event["time"]
            if event["crossed"]:
                complete, bracket = True, event["bracket"]
                event_anchor_T, event_anchor_C = T.copy(), C.copy()
                event_anchor_time, event_anchor_loss = float(t), losses[-1]
                event_lower_T, event_lower_C = event["lower_T"], event["lower_C"]
                event_lower_loss = event_anchor_loss+float(event["lower_diag"][1])
        append_samples(newT, newC, target)
        maxima.append(float(M)); maximum_radii.append(float(rmax))
        logs.append(np.asarray(diag)); means.append(float(diag[0]))
        losses.append(losses[-1]+float(diag[1]))
        T, C, t = newT, newC, float(target)
        if progress and (time.perf_counter()-last_report > 25 or complete):
            print(f"Q4 N={cells}, t={t/3600:.6f} h, R={radius_history[-1]:.8g} m, "
                  f"M={M:.9g}, elapsed={time.perf_counter()-started:.1f} s", flush=True)
            last_report = time.perf_counter()
        if complete:
            break
    means, losses, logs = np.asarray(means), np.asarray(losses), np.asarray(logs)
    water_balance = means+losses-config["initial_C"]
    radius_history = np.asarray(radius_history)
    trace = np.asarray(event_trace, dtype=float).reshape(-1, 4)
    final_radius = float(radius_history[-1])
    lower_time = bracket["t_minus"] if complete else float("nan")
    anchor_radius = float(radius_at(event_anchor_time, radius_data, radius_slopes,
        config["shrinking"], config["radius"])) if complete else float("nan")
    lower_radius = float(radius_at(lower_time, radius_data, radius_slopes,
        config["shrinking"], config["radius"])) if complete else float("nan")
    metadata = {
        **config, "status": "complete" if complete else "incomplete", "completed": complete,
        "finish_time_s": t if complete else None, "finish_time_h": t/3600 if complete else None,
        "last_time_s": t, "start_time_s": 0.0, "cells": cells,
        "mesh": "polynomial10_surface_graded_reference_metres", "bracket": bracket,
        "event_anchor_time_s": event_anchor_time if complete else None,
        "final_radius_m": final_radius, "environment_path": _source_label(environment_path),
        "radius_path": _source_label(radius_path), "source_hashes": hashes, "cache_fingerprint": fingerprint,
        "environment_original_path": str(environment_path), "radius_original_path": str(radius_path),
        "initial_state_source": "Appendix 4 uniform initial state at absolute t=0; no saved Q2/Q3 state",
        "startup_scheme": "min(dt_max,dt_scale*(1+t/time_growth)^2); time_growth=0 gives dt_max",
        "time_fences": "absolute environment and radius knots plus requested outputs and event endpoints",
        "water_balance_definition": "reference-volume mean_C + cumulative reference-normalized boundary loss - initial_C",
        "max_water_balance_residual": float(np.max(np.abs(water_balance))),
        "max_scaled_residual": float(np.max(logs[:, 7])),
        "steps": int(np.sum(logs[:, 2])), "iterations": int(np.sum(logs[:, 3])),
        "rejections": int(np.sum(logs[:, 4])), "damped_iterations": int(np.sum(logs[:, 8])),
        "event_work_steps": event_work_steps, "event_work_iterations": event_work_iterations,
        "discarded_coarse_steps": discarded_coarse_steps, "search_horizons_s": extended_limits,
        "elapsed_seconds": time.perf_counter()-started,
        "physical_output_definition": "fixed radii in metres; exterior values NaN with valid_mask",
        "reference_output_definition": "reference_radii in metres; physical position xi*radius_m",
        "maximum_definition": "all native finite-volume cells plus reconstructed center and moving surface",
        "surface_output_definition": "separately reconstructed actual boundary r=R(t), not the final fixed-radius column",
        "fixed_control_definition": "Appendix 4 from t=0 with constant R0 and the same Q4 time-step rules",
        "termination_note": "strict unrounded max_C < 0.15 at bracket t_plus" if complete else
                            "hard search limit reached; no drying endpoint claimed",
    }
    result = {
        "times": np.asarray(times), "radii": radii, "temperature_K": np.asarray(temperatures),
        "moisture": np.asarray(moistures), "valid_mask": np.asarray(masks, dtype=bool),
        "reference_radii": reference_radii, "xi": xi,
        "temperature_ref_K": np.asarray(reference_temperatures),
        "moisture_ref": np.asarray(reference_moistures),
        "reference_physical_radii": radius_history[:, None]*xi[None, :],
        "radius_m": radius_history, "surface_temperature_K": np.asarray(surface_temperatures),
        "surface_moisture": np.asarray(surface_moistures), "max_C": np.asarray(maxima),
        "max_r": np.asarray(maximum_radii), "mean_C": means, "loss_cumulative": losses,
        "water_balance": water_balance, "logs": logs, "faces": faces, "centers": centers,
        "volumes": volumes, "final_T": T, "final_C": C, "final_radius_m": np.asarray(final_radius),
        "final_physical_faces": (final_radius/config["radius"])*faces,
        "final_physical_centers": (final_radius/config["radius"])*centers,
        "initial_T": initial_T, "initial_C": initial_C, "initial_time_s": np.asarray(0.0),
        "initial_loss_cumulative": np.asarray(0.0), "environment": environment,
        "radius_data": radius_data, "radius_slopes": radius_slopes, "event_trace": trace,
        "event_anchor_T": event_anchor_T, "event_anchor_C": event_anchor_C,
        "event_anchor_time_s": np.asarray(event_anchor_time),
        "event_anchor_loss_cumulative": np.asarray(event_anchor_loss),
        "event_anchor_radius_m": np.asarray(anchor_radius), "event_lower_T": event_lower_T,
        "event_lower_C": event_lower_C, "event_lower_time_s": np.asarray(lower_time),
        "event_lower_loss_cumulative": np.asarray(event_lower_loss),
        "event_lower_radius_m": np.asarray(lower_radius),
        "event_upper_T": T.copy() if complete else np.empty(0),
        "event_upper_C": C.copy() if complete else np.empty(0),
        "event_upper_time_s": np.asarray(t if complete else float("nan")),
        "metadata": metadata,
    }
    save_solution(result, output_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Solve Appendix 4 from t=0 to its strict drying threshold")
    parser.add_argument("--cells", type=int, default=960)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dt-scale", type=float, default=1.0/768.0)
    parser.add_argument("--time-growth", type=float, default=600.0)
    parser.add_argument("--event-dt", type=float, default=.05)
    parser.add_argument("--event-tol", type=float, default=.01)
    parser.add_argument("--fixed-radius", action="store_true")
    parser.add_argument("--sample-count", type=int, default=101)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--radius", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--end-limit", type=float, default=72.0*3600.0)
    parser.add_argument("--hard-end-limit", type=float, default=30.0*24.0*3600.0)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    result = solve_q4(cells=args.cells, dt_max=args.dt, dt_scale=args.dt_scale,
        time_growth=args.time_growth, event_dt=args.event_dt, event_tol=args.event_tol,
        shrinking=not args.fixed_radius, sample_count=args.sample_count,
        environment_path=args.environment, radius_path=args.radius, output_path=args.output,
        end_limit=args.end_limit, hard_end_limit=args.hard_end_limit, recompute=args.recompute)
    print(json.dumps({key: result["metadata"][key] for key in
        ("status", "finish_time_s", "finish_time_h", "final_radius_m", "bracket", "elapsed_seconds")}, indent=2))
    if not result["metadata"]["completed"]:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()
