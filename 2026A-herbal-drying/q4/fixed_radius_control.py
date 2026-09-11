"""Run the accepted Appendix-4 fixed-radius control from the original initial state."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
from q4.solve_q4 import solve_q4


def run_control(*, recompute=False, output_path=None):
    selected = json.loads(
        (bootstrap.ROOT / "validation/q4_selected_config.json").read_text(encoding="utf-8")
    )["fixed_control"]
    result = solve_q4(
        cells=selected["cells"], dt_max=selected["dt_max"],
        dt_scale=selected["dt_scale"], event_dt=selected["event_dt"],
        event_tol=selected["event_tol"], time_growth=selected.get("time_growth", 600.0),
        atol=selected.get("atol"), rtol=selected.get("rtol"),
        shrinking=False, recompute=recompute, output_path=output_path,
    )
    if not result["metadata"]["completed"]:
        raise RuntimeError("The fixed-radius control has not reached its strict drying threshold")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="附录4固定初始半径对照，采用已验收配置")
    parser.add_argument("--recompute", action="store_true", help="从题设初态重新计算")
    parser.add_argument("--output", type=Path, help="独立结果文件；默认 q4/fixed_control.npz")
    args = parser.parse_args()
    solution = run_control(recompute=args.recompute, output_path=args.output)
    print(json.dumps({key: solution["metadata"][key] for key in
        ("finish_time_s", "finish_time_h", "cells", "dt_max", "shrinking")}, indent=2))
