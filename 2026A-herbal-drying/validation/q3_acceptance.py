"""Recompute the selected Q3 acceptance comparisons from saved raw arrays."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
import hashlib
import json
import numpy as np
from common.solver import load_solution
from q3.convergence_test import difference
from validation.q3_checks import LIMITS
from validation.evidence_paths import project_path

ROOT = bootstrap.ROOT


def local_path(value):
    return project_path(value, ROOT)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selected(solution=None, selected_path=None):
    selected_path = local_path(selected_path or ROOT/"validation/q3_selected_config.json")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert selected.get("validated") and selected.get("status") == "validated"
    selected_run_path = local_path(selected["selected_run_path"])
    reference = load_solution(selected_run_path)
    solution = solution if solution is not None else load_solution(ROOT/"q3/solution.npz")
    for key in ("cells", "dt_max", "event_dt"):
        assert solution["metadata"][key] == selected[key] == reference["metadata"][key]
    assert solution["metadata"]["epsilon_t"] == selected["event_tol"]
    assert solution["metadata"]["source_hashes"] == reference["metadata"]["source_hashes"]
    for key in ("times", "radii", "temperature_K", "moisture", "final_T", "final_C",
                "faces", "centers", "volumes", "max_C", "max_r", "logs", "water_balance",
                "loss_cumulative", "event_anchor_T", "event_anchor_C", "event_lower_T",
                "event_lower_C", "event_trace"):
        assert np.array_equal(solution[key], reference[key], equal_nan=True), f"Formal array changed: {key}"
    source_checks = []
    for value, expected in reference["metadata"]["source_hashes"].items():
        path = local_path(value)
        actual = sha256(path)
        assert actual == expected, f"Numerical source or input changed: {path}"
        source_checks.append({"path": path.relative_to(ROOT).as_posix(), "sha256": actual})
    comparisons = []
    assert selected["required_comparisons"], "A selected configuration needs actual refinement evidence"
    for spec in selected["required_comparisons"]:
        left, right = local_path(spec["coarse_path"]), local_path(spec["fine_path"])
        result = difference(load_solution(left), load_solution(right), spec["kind"])
        for key, path in (("coarse_sha256", left), ("fine_sha256", right)):
            if key in spec:
                assert sha256(path) == spec[key], f"Refinement evidence changed: {path}"
        c_limit = float(spec.get("field_limit", LIMITS["field_C"]))
        t_limit = float(spec.get("endpoint_limit_s", result["targets"]["endpoint_s"]))
        result.update(coarse=spec.get("coarse", left.stem), fine=spec.get("fine", right.stem),
                      coarse_path=left.relative_to(ROOT).as_posix(), fine_path=right.relative_to(ROOT).as_posix(),
                      coarse_sha256=sha256(left), fine_sha256=sha256(right),
                      required=True, targets={"full_output_C": c_limit, "endpoint_s": t_limit},
                      C_passed=result["C"]["maximum"] <= c_limit,
                      endpoint_passed=result["finish_time_s"]["difference"] <= t_limit)
        result["passed"] = result["C_passed"] and result["endpoint_passed"]
        comparisons.append(result)
    return {"passed": all(c["passed"] for c in comparisons),
            "selected_config_path": selected_path.relative_to(ROOT).as_posix(),
            "selected_run_path": selected_run_path.relative_to(ROOT).as_posix(),
            "formal_arrays_identical_to_selected_run": True,
            "current_source_and_input_hashes_verified": source_checks,
            "comparisons_recomputed_from_raw_arrays": comparisons,
            "interpretation": "Prespecified refinement differences and reproducibility, not exact-solution error bounds"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"validation/q3_acceptance.json")
    args = parser.parse_args()
    report = validate_selected()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Selected Q3 configuration failed its required refinement comparisons")
