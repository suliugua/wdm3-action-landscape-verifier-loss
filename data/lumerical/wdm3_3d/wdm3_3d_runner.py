#!/usr/bin/env python
"""3D WDM3 verification runner — single CLI entry point for T3D-1 ~ T3D-4.

Usage:
  python -m wdm3_3d.wdm3_3d_runner --task t3d1 [--case C1|C2|C3|C4] [--seed 7]
  python -m wdm3_3d.wdm3_3d_runner --task t3d2 --seed 7 --port 1 [--wl 1500]
  python -m wdm3_3d.wdm3_3d_runner --task t3d3 --seed 7 --port 1 [--wl 1500] [--eps 1e-2,5e-3,1e-3]
  python -m wdm3_3d.wdm3_3d_runner --task t3d4 --seed 7 --port 1 [--wl 1500] [--eps 1e-2]
  python -m wdm3_3d.wdm3_3d_runner --task all   # sequential T3D-1 through T3D-4

Environment:
  LUMOPT_BACKEND=2026  (hardcoded — 3D requires 2026 R1)
"""

import os
import sys

# ── Force 2026 R1 backend before any lumopt imports ──
os.environ["LUMOPT_BACKEND"] = "2026"

import argparse
import json
from pathlib import Path

import lum_backend  # noqa: F401 — verifies backend path
import _lumopt_compat  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="3D WDM3 verification runner (T3D-1 ~ T3D-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wdm3_3d.wdm3_3d_runner --task t3d1 --case C1
  python -m wdm3_3d.wdm3_3d_runner --task t3d2 --seed 7 --port 1 --wl 1500
  python -m wdm3_3d.wdm3_3d_runner --task all --seed 7 --port 1
""",
    )
    parser.add_argument("--task", required=True,
                        choices=["t3d1", "t3d2", "t3d3", "t3d4", "all"],
                        help="Smoke task to run")
    parser.add_argument("--seed", type=int, default=7,
                        help="Warmstart seed (default: 7)")
    parser.add_argument("--port", type=int, default=1,
                        help="Target output port 1/2/3 (default: 1)")
    parser.add_argument("--wl", type=float, default=1550.0,
                        help="Target wavelength in nm (default: 1550)")
    parser.add_argument("--case", default="C1",
                        choices=["C1", "C2", "C3", "C4"],
                        help="T3D-1 test case (default: C1)")
    parser.add_argument("--eps", type=str, default=None,
                        help="Comma-separated eps values for T3D-3 (default: 1e-2,5e-3,1e-3)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="RNG seed for reproducible FD direction")

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    from .wdm3_3d_smoke import (
        t3d1_power_modal_audit,
        t3d2_formal_readout,
        t3d3_directional_fd,
        t3d4_adjoint_validation,
    )

    tasks_to_run = ["t3d1", "t3d2", "t3d3", "t3d4"] if args.task == "all" else [args.task]

    all_results = {}
    all_passed = True

    for task in tasks_to_run:
        try:
            if task == "t3d1":
                r = t3d1_power_modal_audit(
                    case=args.case, out_dir=out_dir, seed_for_c4=args.seed,
                )
            elif task == "t3d2":
                r = t3d2_formal_readout(
                    seed=args.seed, target_port=args.port, wl_nm=args.wl,
                    out_dir=out_dir,
                )
            elif task == "t3d3":
                eps_vals = None
                if args.eps:
                    eps_vals = [float(x.strip()) for x in args.eps.split(",")]
                r = t3d3_directional_fd(
                    seed=args.seed, target_port=args.port, wl_nm=args.wl,
                    eps_values=eps_vals, out_dir=out_dir, rng_seed=args.rng_seed,
                )
            elif task == "t3d4":
                eps = 1e-2
                if args.eps:
                    eps = float(args.eps.split(",")[0])
                r = t3d4_adjoint_validation(
                    seed=args.seed, target_port=args.port, wl_nm=args.wl,
                    eps=eps, out_dir=out_dir, rng_seed=args.rng_seed,
                )

            all_results[task] = r
            if not r.get("passes", False):
                all_passed = False
                if args.task != "all":
                    sys.exit(1)

        except Exception as exc:
            print(f"\n[FATAL] {task} failed: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ── Summary for --task all ──
    if args.task == "all":
        print(f"\n{'='*60}")
        print("ALL TASKS SUMMARY")
        print(f"{'='*60}")
        for task, r in all_results.items():
            status = "PASS" if r.get("passes") else "FAIL"
            wall = r.get("wall_s", 0)
            print(f"  {task}: {status}  ({wall:.0f}s)")
        print(f"\n  overall: {'PASS' if all_passed else 'FAIL'}")
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
