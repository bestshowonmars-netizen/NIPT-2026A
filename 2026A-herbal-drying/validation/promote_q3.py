"""Promote an already computed, verified fine Q3 run to the formal output.

No numerical work is represented as a new simulation: the exact stored run is
copied, and the choice is recorded with raw-array refinement comparisons.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import json
import shutil
from common.solver import load_solution
from q3.convergence_test import difference
from validation.q3_checks import check_solution, LIMITS
from validation.evidence_paths import project_path

ROOT = bootstrap.ROOT


def promote(dt=.03125):
    selected_name = f"q3_N960_dt{dt:g}_production_evdt0.05_tol0.01"
    selected_run = ROOT/"q3/runs"/(selected_name+".npz")
    solution = load_solution(selected_run)
    check = check_solution(solution, load_solution(ROOT/"q2/solution.npz"), require_dense=True)
    required = []
    for matrix in ("time", "space", "event", "event_step", "picard"):
        report_path = ROOT/"validation"/f"q3_convergence_{matrix}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = {record["label"]: record for record in report["runs"]}
        chosen = [entry for entry in report["comparisons"]
                  if matrix != "time" or entry["fine"] == selected_name]
        assert chosen, f"Missing required comparison for {matrix}"
        for entry in chosen:
            left, right = (project_path(records[entry[side]]["path"], ROOT) for side in ("coarse", "fine"))
            actual = difference(load_solution(left), load_solution(right), entry["kind"])
            assert actual["passed"], f"Required comparison failed: {entry['coarse']} -> {entry['fine']}"
            required.append({"kind": entry["kind"], "coarse": entry["coarse"], "fine": entry["fine"],
                             "coarse_path": left.relative_to(ROOT).as_posix(),
                             "fine_path": right.relative_to(ROOT).as_posix(),
                             "coarse_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
                             "fine_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
                             "field_limit": actual["targets"]["full_output_C"],
                             "endpoint_limit_s": actual["targets"]["endpoint_s"],
                             "record_path": report_path.relative_to(ROOT).as_posix()})
    for matrix in ("event", "event_step"):
        report = json.loads((ROOT/"validation"/f"q3_convergence_{matrix}.json").read_text(encoding="utf-8"))
        for record in report["runs"]:
            assert project_path(record["base_path"], ROOT).resolve() == selected_run.resolve(), "Event checks must use the selected run's anchor"
    metadata = solution["metadata"]
    selected = {"status": "validated", "validated": True, "cells": metadata["cells"],
                "dt_max": metadata["dt_max"], "event_dt": metadata["event_dt"],
                "event_tol": metadata["epsilon_t"], "finish_time_s": metadata["finish_time_s"],
                "finish_time_h": metadata["finish_time_h"],
                "selected_run_label": selected_name,
                "selected_run_path": selected_run.relative_to(ROOT).as_posix(),
                "predeclared_limits": LIMITS, "required_comparisons": required,
                "selection_reason": "Use the finer member of the first tested time pair satisfying the predeclared endpoint and field criteria; native-grid spatial, event, seed and Picard controls also pass",
                "validation_boundary": "Separate observed refinement differences, not an exact-solution total error bound",
                "formalization": "Byte-for-byte promotion of the completed verified run; no new simulation claimed"}
    shutil.copy2(selected_run, ROOT/"q3/solution.npz")
    shutil.copy2(selected_run.with_suffix(".json"), ROOT/"q3/solution.json")
    (ROOT/"validation/q3_selected_config.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "required_comparisons": len(required),
                      "finish_time_s": metadata["finish_time_s"], "full_state_checks_passed": check["passed"],
                      "formal_arrays_copied_without_change": True}, ensure_ascii=False, indent=2))
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=.03125)
    args = parser.parse_args()
    promote(args.dt)
