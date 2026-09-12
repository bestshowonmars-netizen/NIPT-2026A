"""Verify the fresh Q4 rerun and preserved controls without long new solves."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import bootstrap
import argparse
import datetime
import hashlib
import json
from unittest.mock import patch
import numpy as np
from common.solver import load_solution
from validation.q4_checks import check_solution, core_checks
from q4.convergence_test import difference
from validation.evidence_paths import project_path

OUTPUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, report):
    (OUTPUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def unchanged_originals():
    launch = json.loads((OUTPUT / "launch.json").read_text(encoding="utf-8"))
    for name, expected in launch["frozen_inputs_and_original_outputs_sha256"].items():
        assert digest(ROOT / name) == expected, f"Original file changed: {name}"
    return launch


def preflight():
    unchanged_originals()
    # Keep the existing restart test's temporary state inside this rerun folder.
    scratch = OUTPUT / "scratch"
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)
    with patch.object(bootstrap, "RUNTIME", scratch):
        core = core_checks()
    selected = json.loads((ROOT / "validation/q4_selected_config.json").read_text(encoding="utf-8"))
    fixed_path = ROOT / "q4/fixed_control.npz"
    fixed_source = project_path(selected["fixed_control"]["formal_case"], ROOT)
    assert digest(fixed_path) == digest(fixed_source)
    fixed = load_solution(fixed_path)
    checks = check_solution(fixed)
    assert fixed["metadata"]["shrinking"] is False
    comparisons, evidence = [], {}
    for requirement in selected["required_comparisons"]:
        if requirement["scope"] != "fixed":
            continue
        report_path = ROOT / requirement["report"]
        saved = json.loads(report_path.read_text(encoding="utf-8"))["comparisons"][requirement.get("index", 0)]
        paths = {side: project_path(saved[side + "_path"], ROOT) for side in ("coarse", "fine")}
        for side, path in paths.items():
            assert digest(path) == saved[side + "_sha256"]
            evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        result = difference(load_solution(paths["coarse"]), load_solution(paths["fine"]), saved["kind"],
                            paths["coarse"].relative_to(ROOT).as_posix(), paths["fine"].relative_to(ROOT).as_posix())
        assert result["passed"] and saved["passed"]
        assert result["targets"] == saved["targets"]
        assert result["C"]["maximum"] == saved["C"]["maximum"]
        assert result["finish_time_s"]["difference"] == saved["finish_time_s"]["difference"]
        comparisons.append(result)
        evidence[requirement["report"]] = digest(report_path)
    assert {item["kind"] for item in comparisons} == {"fixed_time", "fixed_space"}
    report = {"passed": True, "core_checks": core,
              "fixed_control": {"recomputed_full_trajectory": False, "checks": checks,
                  "path": fixed_path.relative_to(ROOT).as_posix(), "sha256": digest(fixed_path),
                  "formal_case_bytes_identical": True, "finish_time_s": fixed["metadata"]["finish_time_s"],
                  "comparisons_recomputed_from_saved_arrays": comparisons, "evidence_sha256": evidence},
              "original_files_unchanged": True}
    write("preflight.json", report)
    print(json.dumps({"preflight_passed": True, "fixed_control_reused_after_hash_and_evidence_checks": True,
                      "core": core}, ensure_ascii=False, indent=2), flush=True)
    return report


def verify():
    launch = unchanged_originals()
    assert launch["status"] == "computed_pending_checks"
    pre = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    assert pre["passed"]
    fresh = load_solution(OUTPUT / "solution.npz")
    original = load_solution(ROOT / "q4/solution.npz")
    checks = check_solution(fresh)
    fresh_keys = {key for key, value in fresh.items() if isinstance(value, np.ndarray)}
    old_keys = {key for key, value in original.items() if isinstance(value, np.ndarray)}
    assert fresh_keys == old_keys
    comparisons = {}
    for key in sorted(fresh_keys):
        a, b = fresh[key], original[key]
        assert a.dtype == b.dtype and a.shape == b.shape, key
        equal = bool(np.array_equal(a, b, equal_nan=True))
        finite = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(abs(a[finite].astype(float) - b[finite].astype(float)))) if np.any(finite) else 0.
        comparisons[key] = {"shape": list(a.shape), "dtype": str(a.dtype),
                            "array_equal_including_nan": equal, "maximum_finite_difference": maximum,
                            "nan_masks_identical": bool(np.array_equal(np.isnan(a), np.isnan(b)))}
    write("array_comparison.json", {"arrays": comparisons,
          "all_equal_including_nan": all(item["array_equal_including_nan"] for item in comparisons.values())})
    assert all(item["array_equal_including_nan"] for item in comparisons.values()), \
        "Fresh arrays differ from the original; inspect array_comparison.json"
    fm, om = fresh["metadata"], original["metadata"]
    assert fm["finish_time_s"] == om["finish_time_s"] and fm["bracket"] == om["bracket"]
    assert fm["source_hashes"] == om["source_hashes"]
    assert fm["start_time_s"] == 0 and np.all(fresh["initial_T"] == 301.15) and np.all(fresh["initial_C"] == 2.55)
    differences = {key: {"fresh": fm.get(key), "original": om.get(key)}
                   for key in sorted(fm.keys() | om.keys()) if fm.get(key) != om.get(key)}
    crossings = np.nonzero(fresh["mean_C"] < .15)[0]
    assert len(crossings) and crossings[0] > 0
    hi = int(crossings[0]); lo = hi - 1
    assert fresh["mean_C"][lo] >= .15 and np.all(fresh["mean_C"][:hi] >= .15)
    mean_bracket = {"threshold": .15, "criterion": "mean_C < 0.15; comparison only, not the formal all-domain criterion",
                    "t_minus_s": float(fresh["times"][lo]), "t_plus_s": float(fresh["times"][hi]),
                    "mean_C_minus": float(fresh["mean_C"][lo]), "mean_C_plus": float(fresh["mean_C"][hi]),
                    "global_max_C_minus": float(fresh["max_C"][lo]), "global_max_C_plus": float(fresh["max_C"][hi]),
                    "width_s": float(fresh["times"][hi] - fresh["times"][lo]),
                    "advance_vs_formal_endpoint_interval_h": [float((fm["finish_time_s"] - fresh["times"][hi])/3600),
                                                                float((fm["finish_time_s"] - fresh["times"][lo])/3600)],
                    "method": "First threshold-crossing pair among saved 60 s outputs; no interpolated or sub-minute root claimed"}
    write("mean_threshold_bracket.json", mean_bracket)
    report = {"passed": True, "question": 4, "fresh_solve_from_initial_state": True,
              "cached_solution_used_to_initialize": False, "configuration": launch["parameters"],
              "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "finish_time_s": fm["finish_time_s"], "finish_time_h": fm["finish_time_h"],
              "strict_final_max_C": float(fresh["max_C"][-1]), "strict_final_max_r_m": float(fresh["max_r"][-1]),
              "final_radius_m": float(fresh["radius_m"][-1]), "bracket": fm["bracket"],
              "fresh_solution_sha256": digest(OUTPUT / "solution.npz"),
              "original_solution_sha256": digest(ROOT / "q4/solution.npz"),
              "array_count": len(comparisons), "all_numeric_arrays_equal_including_nan": True,
              "array_comparisons": comparisons, "metadata_differences": differences,
              "solution_checks": checks, "mean_threshold_saved_bracket": mean_bracket,
              "preflight_path": "preflight.json", "preflight_sha256": digest(OUTPUT / "preflight.json"),
              "core_checks_passed": pre["core_checks"]["passed"],
              "fixed_control_evidence_retained_and_verified": True,
              "original_files_unchanged": True, "original_files_sha256": launch["frozen_inputs_and_original_outputs_sha256"],
              "interpretation": "A fresh execution of the same accepted numerical model plus deterministic consistency checks; not experimental physical validation"}
    unchanged_originals()
    write("receipt.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "finish_time_s", "finish_time_h", "array_count",
                      "all_numeric_arrays_equal_including_nan", "strict_final_max_C", "mean_threshold_saved_bracket")},
                     ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else verify()
