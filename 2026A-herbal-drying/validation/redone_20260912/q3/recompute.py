"""Recompute Q3 from the full Q2 endpoint without changing historical results."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from common.solver import load_solution
from common.continuation import advance_interval
from common.data_loader import load_environment
from q3.solve_q3 import solve_q3
from validation.q3_checks import check_solution


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.log = path.open("w", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, OUT / "progress.log")
    started = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    source = ROOT / "q2/solution.npz"
    old_path = ROOT / "q3/solution.npz"
    retained = [source, ROOT / "q2/final_state.npz", old_path,
                ROOT / "submission_results/result3.xlsx"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in retained}
    write_json("started.json", {"started_at_utc": started_at, "recompute": True,
        "source": str(source), "source_final_state": str(ROOT / "q2/final_state.npz"),
        "output": str(OUT / "solution.npz"), "cells": 960,
        "dt_max_s": 1 / 32, "event_dt_s": 1 / 32, "event_tolerance_s": .01,
        "retained_sha256_before": before})
    new = solve_q3(cells=960, dt_max=1 / 32, event_dt=1 / 32, event_tol=.01,
        source_path=source, output_path=OUT / "solution.npz", recompute=True, progress=True)
    print("Q3 recomputation complete; checking saved states and conservation.", flush=True)
    saved = load_solution(OUT / "solution.npz")
    q2 = load_solution(source)
    checks = check_solution(saved, q2, require_dense=True)
    old = load_solution(old_path)
    arrays = {}
    assert set(saved) == set(old)
    for key in sorted(set(saved) - {"metadata"}):
        a, b = np.asarray(saved[key]), np.asarray(old[key])
        same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        delta = None
        if a.shape == b.shape:
            mask = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[mask] - b[mask]))) if np.any(mask) else 0.0
        arrays[key] = {"shape": list(a.shape), "exact_equal_including_nan": bool(same),
                       "maximum_finite_absolute_difference": delta}
    # Check a genuine save/load continuation using the same accepted time schedule.
    env = load_environment(ROOT / "data/attachment1.xlsx")
    m = saved["metadata"]
    args = (1 / 32, env, q2["faces"], q2["centers"], q2["volumes"],
            m["h"], m["hm"], m["atol"], m["rtol"], m["max_iterations"])
    T0, C0 = q2["final_T"], q2["final_C"]
    Ta, Ca, da = advance_interval(T0.copy(), C0.copy(), 10800., 10920., *args)
    Ts, Cs, ds = advance_interval(T0.copy(), C0.copy(), 10800., 10860., *args)
    np.savez_compressed(OUT / "restart_state.npz", T=Ts, C=Cs, time_s=10860.)
    with np.load(OUT / "restart_state.npz", allow_pickle=False) as state:
        Tr, Cr, dr = advance_interval(state["T"], state["C"], float(state["time_s"]), 10920., *args)
    restart = {"temperature_K": float(np.max(np.abs(Ta - Tr))),
               "moisture": float(np.max(np.abs(Ca - Cr))),
               "continued_loss": float(abs(da[1] - ds[1] - dr[1]))}
    assert max(restart.values()) <= 1e-12
    times, means = saved["times"], saved["mean_C"]
    index = int(np.flatnonzero(means < .15)[0])
    mean_interval = [float(times[index-1]), float(times[index])]
    finish = float(m["finish_time_s"])
    after = {str(p.relative_to(ROOT)): digest(p) for p in retained}
    assert before == after
    kernel_hashes_unchanged = all(digest(p) == value for p, value in m["source_hashes"].items())
    assert kernel_hashes_unchanged
    same_arrays = all(row["exact_equal_including_nan"] for row in arrays.values())
    receipt = {"status": "complete", "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_elapsed_seconds_including_checks": time.perf_counter() - started,
        "fresh_forward_recompute": True, "cache_reuse": False,
        "source": str(source), "output": str(OUT / "solution.npz"),
        "cells": 960, "dt_max_s": 1 / 32, "event_dt_s": 1 / 32,
        "event_tolerance_s": .01, "finish_time_s": finish,
        "finish_time_h": finish / 3600, "event_bracket": m["bracket"],
        "global_maximum_final_C": float(saved["max_C"][-1]),
        "global_maximum_final_r_m": float(saved["max_r"][-1]),
        "mean_C_final": float(means[-1]),
        "surface_C_final": float(saved["moisture"][-1, -1]),
        "loss_cumulative_final": float(saved["loss_cumulative"][-1]),
        "fresh_solution_checks": checks, "save_reload_restart_maximum_errors": restart,
        "same_numeric_arrays_as_old_formal": same_arrays,
        "old_formal_numeric_array_comparison": arrays,
        "old_formal_finish_difference_s": finish - float(old["metadata"]["finish_time_s"]),
        "retained_sha256_before": before, "retained_sha256_after": after,
        "source_and_kernel_hashes_unchanged": kernel_hashes_unchanged,
        "new_solution_sha256": digest(OUT / "solution.npz"),
        "mean_threshold_saved_bracket_s": mean_interval,
        "mean_C_at_bracket": [float(means[index-1]), float(means[index])],
        "mean_threshold_saved_bracket_h": [v / 3600 for v in mean_interval],
        "mean_criterion_early_stop_hours_bracket": [(finish - mean_interval[1]) / 3600,
                                                     (finish - mean_interval[0]) / 3600],
        "mean_threshold_note": "Adjacent saved 60-second states; no interpolated or reintegrated mean-threshold event claimed.",
        "interpretation": "Fresh computation and numerical verification of the specified effective model; not experimental validation."}
    write_json("receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "finish_time_s", "finish_time_h",
        "same_numeric_arrays_as_old_formal", "event_bracket", "mean_threshold_saved_bracket_h")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        error = traceback.format_exc()
        write_json("failure.json", {"failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "traceback": error})
        print(error, flush=True)
        raise
