"""Question-three continuation from a saved, unrounded question-two solution."""
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
from common.continuation import advance_interval, global_maximum, sample_fields
from q3.config import make_config
from q3.end_time_detector import refine_endpoint

ROOT = bootstrap.ROOT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(config, cells, source_path, final_state_path):
    paths = [source_path, ROOT / "data/attachment1.xlsx"]
    if final_state_path is not None:
        paths.append(final_state_path)
    paths.extend(ROOT / name for name in (
        "common/solver.py", "common/finite_volume.py", "common/continuation.py",
        "common/material_properties.py", "common/boundary_conditions.py",
        "common/interpolation.py", "common/data_loader.py", "q2/config.py",
        "q3/config.py", "q3/end_time_detector.py", "q3/solve_q3.py"))
    hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    payload = json.dumps({"config": config, "cells": cells, "files": hashes},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), hashes


def _load_start(source_path, final_state_path, cells, config, environment):
    source = load_solution(source_path)
    meta = source["metadata"]
    start = config["start_time"]
    if int(meta.get("question", -1)) != 2 or int(meta.get("cells", -1)) != cells:
        raise ValueError("Q3 requires a question-two source with the requested native cell count")
    for key in ("radius", "h", "hm", "initial_T", "initial_C"):
        if meta.get(key) != config[key]:
            raise ValueError(f"Question-two source has incompatible {key}")
    source_times = np.asarray(source["times"], dtype=float)
    if not np.array_equal(source_times, np.arange(int(start) + 1, dtype=float)):
        raise ValueError("Q2 prefix must contain every integer second from 0 through 10800")
    faces = np.asarray(source["faces"], dtype=float)
    centers = np.asarray(source["centers"], dtype=float)
    volumes = np.asarray(source["volumes"], dtype=float)
    T = np.asarray(source["final_T"], dtype=float).copy()
    C = np.asarray(source["final_C"], dtype=float).copy()
    radii = np.asarray(source["radii"], dtype=float)
    if (T.shape != (cells,) or C.shape != (cells,) or faces.shape != (cells + 1,)
            or centers.shape != (cells,) or volumes.shape != (cells,)):
        raise ValueError("Q2 source does not contain full native finite-volume states")
    if not (np.all(np.diff(faces) > 0) and np.all(volumes > 0)
            and faces[0] == 0 and faces[-1] == config["radius"]):
        raise ValueError("Invalid source geometry")
    if (len(radii) < 21 or (len(radii) - 1) % 20 != 0
            or not np.allclose(radii, np.linspace(0, config["radius"], len(radii)), rtol=0, atol=1e-14)):
        raise ValueError("Source sampling must contain the 21 required radial locations")
    if not (np.isfinite(T).all() and np.isfinite(C).all() and np.min(T) > 0 and np.min(C) > 0):
        raise ValueError("Invalid source temperature or moisture state")
    if final_state_path is not None:
        with np.load(final_state_path, allow_pickle=False) as state:
            if float(state["time_s"]) != start:
                raise ValueError("The production continuation state is not at 10800 seconds")
            final_cfg = json.loads(str(state["config_json"]))
            if int(final_cfg.get("question", -1)) != 2 or int(final_cfg.get("cells", -1)) != cells:
                raise ValueError("The production final-state configuration is incompatible")
            for key, expected in (("faces", faces), ("centers", centers), ("volumes", volumes),
                                  ("temperature_K", T), ("moisture", C), ("environment", environment)):
                if not np.array_equal(state[key], expected):
                    raise ValueError(f"Q2 final_state and source/environment disagree: {key}")
            # Read the full state from the explicitly supplied continuation artifact.
            T, C = state["temperature_K"].copy(), state["moisture"].copy()
    for name in ("temperature_K", "moisture"):
        if source[name].shape != (len(source_times), len(radii)) or not np.isfinite(source[name]).all():
            raise ValueError(f"Invalid Q2 prefix {name}")
    if source["logs"].shape != (int(start), 9) or not np.isfinite(source["logs"]).all():
        raise ValueError("Q2 prefix requires complete nine-column per-second diagnostics")
    sampled_T, sampled_C = sample_fields(T, C, start, environment, faces, centers, radii,
                                         config["h"], config["hm"])
    if (not np.allclose(sampled_T, source["temperature_K"][-1], rtol=0, atol=1e-9)
            or not np.allclose(sampled_C, source["moisture"][-1], rtol=0, atol=1e-11)):
        raise ValueError("The Q2 sampled endpoint and its full continuation state disagree")
    return source, T, C, faces, centers, volumes, radii


def _aggregate_logs(block):
    diag = np.zeros(9)
    diag[0] = block[-1, 0]
    for column in (1, 2, 3, 4, 8):
        diag[column] = np.sum(block[:, column])
    active = block[:, 2] > 0
    diag[5] = np.min(block[active, 5]) if np.any(active) else 0.0
    diag[6] = np.max(block[:, 6])
    diag[7] = np.max(block[:, 7])
    return diag


def solve_q3(cells=960, dt_max=1.0, event_dt=0.05, event_tol=0.01,
             source_path=None, output_path=None, recompute=False, progress=True,
             end_limit=72.0 * 3600.0, hard_end_limit=30.0 * 24.0 * 3600.0,
             atol=None, rtol=None):
    """Continue native Q2 states; an exhausted hard limit returns status='incomplete'."""
    started = time.perf_counter()
    config = make_config(dt_max, event_dt, event_tol, end_limit, hard_end_limit, atol, rtol)
    cells = int(cells)
    default_source = ROOT / "q2/solution.npz"
    source_path = Path(source_path).resolve() if source_path is not None else default_source
    final_state_path = ROOT / "q2/final_state.npz" if source_path.resolve() == default_source.resolve() else None
    output_path = Path(output_path) if output_path is not None else ROOT / "q3/solution.npz"
    if not source_path.exists() or (final_state_path is not None and not final_state_path.exists()):
        raise FileNotFoundError("The required unrounded Q2 source/final_state is missing")
    fingerprint, hashes = _fingerprint(config, cells, source_path, final_state_path)
    if output_path.exists() and not recompute:
        cached = load_solution(output_path)
        if cached["metadata"].get("cache_fingerprint") == fingerprint:
            if progress:
                print(f'Q3 cache: {cached["metadata"].get("status")}, {output_path}', flush=True)
            return cached
    environment = load_environment(ROOT / "data/attachment1.xlsx")
    source, T, C, faces, centers, volumes, radii = _load_start(
        source_path, final_state_path, cells, config, environment)
    start = config["start_time"]
    spacing = int(config["output_interval"])
    indices = np.arange(0, int(start) + 1, spacing, dtype=int)
    times = list(source["times"][indices].astype(float))
    temperatures = list(source["temperature_K"][indices].copy())
    moistures = list(source["moisture"][indices].copy())
    means = [config["initial_C"]]
    losses = [0.0]
    logs = [np.zeros(9)]
    for left, right in zip(indices[:-1], indices[1:]):
        diag = _aggregate_logs(source["logs"][left:right])
        logs.append(diag)
        means.append(float(diag[0]))
        losses.append(losses[-1] + float(diag[1]))
    maxima, max_r = [float("nan")] * len(times), [float("nan")] * len(times)
    M, rmax, _, _ = global_maximum(T, C, start, environment, faces, centers, config["h"], config["hm"])
    maxima[-1], max_r[-1] = float(M), float(rmax)
    if M < config["threshold"]:
        raise ValueError("Q2 is already below threshold; Q3 cannot locate the first crossing from this seed")
    t = start
    horizon = config["initial_end_limit"]
    extended_limits = [horizon]
    event_trace = []
    event_anchor_T, event_anchor_C = np.empty(0), np.empty(0)
    event_lower_T, event_lower_C = np.empty(0), np.empty(0)
    event_anchor_time = event_anchor_loss = event_lower_loss = float("nan")
    bracket = None
    complete = False
    event_work_steps = event_work_iterations = 0
    discarded_coarse_steps = 0
    last_report = started
    if progress:
        print(f"Q3 start: N={cells}, t={start:g} s, dt<={dt_max:g} s, M={M:.9g}", flush=True)
    while t < config["hard_end_limit"] - 1e-10:
        target = min((np.floor(t / spacing) + 1.0) * spacing, config["hard_end_limit"])
        while target > horizon and horizon < config["hard_end_limit"]:
            horizon = min(config["hard_end_limit"], horizon * config["horizon_growth"])
            extended_limits.append(horizon)
            if progress:
                print(f"Q3 remains undried; extend search to {horizon / 3600:g} h", flush=True)
        newT, newC, diag = advance_interval(
            T, C, t, target, config["dt_max"], environment, faces, centers, volumes,
            config["h"], config["hm"], config["atol"], config["rtol"], config["max_iterations"])
        M, rmax, _, _ = global_maximum(newT, newC, target, environment, faces, centers,
                                      config["h"], config["hm"])
        if M < config["threshold"]:
            event = refine_endpoint(T, C, t, target, environment, faces, centers, volumes, config)
            event_trace.extend(event["trace"].tolist())
            event_work_steps += event["work_steps"]
            event_work_iterations += event["work_iterations"]
            discarded_coarse_steps += int(diag[2])
            newT, newC, diag = event["T"], event["C"], event["diag"]
            M, rmax, target = event["M"], event["r"], event["time"]
            if event["crossed"]:
                complete, bracket = True, event["bracket"]
                event_anchor_T, event_anchor_C = T.copy(), C.copy()
                event_anchor_time, event_anchor_loss = t, losses[-1]
                event_lower_T, event_lower_C = event["lower_T"], event["lower_C"]
                event_lower_loss = event_anchor_loss + float(event["lower_diag"][1])
        sampleT, sampleC = sample_fields(newT, newC, target, environment, faces, centers,
                                         radii, config["h"], config["hm"])
        if not target > times[-1]:
            raise RuntimeError("Continuation output times must increase without duplicate 10800-second rows")
        times.append(float(target))
        temperatures.append(sampleT)
        moistures.append(sampleC)
        maxima.append(float(M))
        max_r.append(float(rmax))
        logs.append(np.asarray(diag))
        means.append(float(diag[0]))
        losses.append(losses[-1] + float(diag[1]))
        T, C, t = newT, newC, float(target)
        if progress and (time.perf_counter() - last_report > 25 or complete):
            print(f"Q3 N={cells}, t={t / 3600:.6f} h, M={M:.9g}, elapsed={time.perf_counter()-started:.1f} s", flush=True)
            last_report = time.perf_counter()
        if complete:
            break
    means, losses, logs = np.asarray(means), np.asarray(losses), np.asarray(logs)
    water_balance = means + losses - config["initial_C"]
    continuation_logs = logs[len(indices):]
    metadata = {
        **config, "question": 3, "model_question": 2,
        "status": "complete" if complete else "incomplete", "completed": complete,
        "finish_time_s": t if complete else None, "finish_time_h": t / 3600 if complete else None,
        "last_time_s": t, "start_time_s": start, "cells": cells, "sample_count": len(radii),
        "mesh": source["metadata"].get("mesh"), "bracket": bracket,
        "event_anchor_time_s": event_anchor_time if complete else None,
        "source_path": str(source_path.resolve()), "source_metadata": source["metadata"],
        "source_hashes": hashes, "cache_fingerprint": fingerprint,
        "seed_loss_cumulative": float(losses[len(indices)-1]),
        "max_water_balance_residual": float(np.max(np.abs(water_balance))),
        "max_scaled_residual": float(np.max(logs[:, 7])),
        "steps": int(np.sum(continuation_logs[:, 2])),
        "iterations": int(np.sum(continuation_logs[:, 3])),
        "rejections": int(np.sum(continuation_logs[:, 4])),
        "damped_iterations": int(np.sum(continuation_logs[:, 8])),
        "event_work_steps": event_work_steps, "event_work_iterations": event_work_iterations,
        "discarded_coarse_steps": discarded_coarse_steps,
        "search_horizons_s": extended_limits, "elapsed_seconds": time.perf_counter() - started,
        "maximum_definition": "all native finite-volume cells plus reconstructed center and surface",
        "prefix_maximum_note": "max_C/max_r unavailable before 10800 s; Q2 retained sampled history only",
        "termination_note": "strict unrounded max_C < 0.15 at bracket t_plus" if complete else "hard search limit reached; no drying endpoint claimed",
    }
    result = {"times": np.asarray(times), "radii": radii, "temperature_K": np.asarray(temperatures),
              "moisture": np.asarray(moistures), "max_C": np.asarray(maxima), "max_r": np.asarray(max_r),
              "mean_C": means, "loss_cumulative": losses, "water_balance": water_balance,
              "logs": logs, "final_T": T, "final_C": C, "faces": faces, "centers": centers,
              "volumes": volumes, "event_trace": np.asarray(event_trace, dtype=float).reshape(-1, 4),
              "event_anchor_T": event_anchor_T, "event_anchor_C": event_anchor_C,
              "event_anchor_time_s": np.asarray(event_anchor_time),
              "event_anchor_loss_cumulative": np.asarray(event_anchor_loss),
              "event_lower_T": event_lower_T, "event_lower_C": event_lower_C,
              "event_lower_loss_cumulative": np.asarray(event_lower_loss),
              "metadata": metadata}
    save_solution(result, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Continue Q2 to the strict Q3 drying threshold")
    parser.add_argument("--cells", type=int, default=960)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--event-dt", type=float, default=0.05)
    parser.add_argument("--event-tol", type=float, default=0.01)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--end-limit", type=float, default=72.0 * 3600.0)
    parser.add_argument("--hard-end-limit", type=float, default=30.0 * 24.0 * 3600.0)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = solve_q3(cells=args.cells, dt_max=args.dt, event_dt=args.event_dt, event_tol=args.event_tol,
                      source_path=args.source, output_path=args.output, recompute=args.recompute,
                      end_limit=args.end_limit, hard_end_limit=args.hard_end_limit)
    print(json.dumps({key: result["metadata"][key] for key in
                      ("status", "finish_time_s", "finish_time_h", "bracket", "elapsed_seconds")}, indent=2))
    if not result["metadata"]["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
