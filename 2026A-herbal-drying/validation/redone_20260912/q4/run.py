"""Explicit fresh Q4 rerun; all new evidence stays beside this driver."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import bootstrap
import datetime
import hashlib
import json
import traceback
from q4.solve_q4 import solve_q4

OUTPUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.original.flush()
        self.log.flush()


def main():
    target = OUTPUT / "solution.npz"
    if target.exists() or (OUTPUT / "launch.json").exists():
        raise FileExistsError("This rerun already has evidence; preserve it instead of overwriting")
    metadata = json.loads((ROOT / "q4/solution.json").read_text(encoding="utf-8"))
    frozen = {name: digest(ROOT / name) for name in metadata["source_hashes"]}
    assert frozen == metadata["source_hashes"], "An original numerical source or input changed"
    for name in ("q4/solution.npz", "q4/solution.json", "q4/fixed_control.npz",
                 "q4/fixed_control.json", "validation/q4_selected_config.json"):
        frozen[name] = digest(ROOT / name)
    parameters = dict(cells=1280, dt_max=1/32, dt_scale=1/768,
                      time_growth=600., event_dt=1/32, event_tol=.01,
                      shrinking=True, recompute=True, progress=True)
    launch = {"status": "running", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "parameters": parameters, "output_path": target.relative_to(ROOT).as_posix(),
              "initialization": "Fresh Appendix 4 uniform initial fields at absolute t=0; no Q2/Q3 seed",
              "frozen_inputs_and_original_outputs_sha256": frozen}
    (OUTPUT / "launch.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "solve.log").open("x", encoding="utf-8") as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_stdout, log), Tee(old_stderr, log)
        try:
            result = solve_q4(output_path=target, **parameters)
            assert result["metadata"]["completed"], "Fresh Q4 did not reach the strict criterion"
            launch.update(status="computed_pending_checks", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          finish_time_s=result["metadata"]["finish_time_s"],
                          fresh_solution_sha256=digest(target))
            print(json.dumps({key: result["metadata"][key] for key in
                              ("finish_time_s", "finish_time_h", "bracket", "elapsed_seconds")}, indent=2), flush=True)
        except BaseException:
            launch["status"] = "failed"
            traceback.print_exc()
            raise
        finally:
            (OUTPUT / "launch.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    main()
