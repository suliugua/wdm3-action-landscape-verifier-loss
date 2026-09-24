#!/usr/bin/env python3
"""Final batch fresh-sim. Skips existing fresh_sim.json. Auto-resume."""
import sys, os, json, time
import numpy as np
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)); PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR); sys.path.insert(0, PARENT_DIR)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ["LUMOPT_BACKEND"] = "2026"
import lum_backend, _lumopt_compat  # noqa
from run_pilot import Session
from metadata import make_metadata
from wdm3_3d.wdm3_3d_device import load_warmstart_params

BASE = os.path.join(PROJECT_DIR, "results", "final_rerun_2026_08_13_protocol_v2")

# (anchor, arm, repeat, accept_types)
JOBS = [
    ("seed7",  "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed2",  "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed3",  "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed4",  "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed1",  "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed13", "unified_protocol", 0, {"wu_a", "probe_a", "bb_a", "bs_a"}),
    ("seed3",  "sign_static",      0, {"sa_a", "ss_a"}),
    ("seed4",  "sign_static",      0, {"sa_a", "ss_a"}),
    ("seed2",  "sign_static",      0, {"sa_a", "ss_a"}),
    ("seed10", "null_banded",      0, {"warmup_accept", "banded_band_accept", "banded_spatial_accept"}),
    ("seed9",  "sign_banded",      0, {"wu_a", "bb_a", "sb_a"}),
    ("seed1",  "random_dbs",       1, {"dbs_accept"}),
    ("seed13", "random_dbs",       1, {"dbs_accept"}),
]

for anchor, arm, repeat, accept_types in JOBS:
    seed_num = int(anchor.replace("seed", ""))
    arm_dir = os.path.join(BASE, anchor, arm, f"budget200", f"repeat{repeat}")
    traj_path = os.path.join(arm_dir, "trajectory.json")
    fresh_path = os.path.join(arm_dir, "fresh_sim.json")

    if os.path.exists(fresh_path):
        print(f"SKIP {anchor}/{arm}/r{repeat}: fresh exists")
        continue
    if not os.path.exists(traj_path):
        print(f"SKIP {anchor}/{arm}/r{repeat}: no trajectory")
        continue

    with open(traj_path) as f:
        traj = json.load(f)

    wb = (np.asarray(load_warmstart_params(seed_num), dtype=float).reshape(-1) > 0.5).astype(float)
    params = wb.copy(); n = 0
    for step in traj["steps"]:
        if step["type"] not in accept_types:
            continue
        if "indices" in step:
            pixels = step["indices"]
        elif "idx" in step:
            pixels = [step["idx"]]
        else:
            continue
        for p in pixels:
            params[p] = 1.0 - params[p]
        n += 1

    online = traj.get("final_cumulative_dJ", traj["steps"][-1].get("cumulative_dJ"))

    print(f"\n{anchor}/{arm}/r{repeat}: {n} accepts, online={online:+.6e}")
    t0 = time.time()
    with Session(anchor, tag=f"bl_{anchor}") as s:
        bl = s.fom
    print(f"  baseline={bl:.6e} ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    with Session(anchor, tag=f"fin_{anchor}_{arm}", params=params) as s:
        fom = s.fom
    wall = time.time() - t1
    fd = fom - bl
    drift = fd - online
    ret = fd / online if abs(online) > 1e-15 else None

    with open(fresh_path, "w") as f:
        json.dump(dict(anchor=anchor, arm=arm, repeat=repeat,
                        fresh_dJ=fd, online_final_dJ=online, drift_abs=drift,
                        retention=ret, n_accepts=n, wall_clock_s=wall,
                        metadata=make_metadata(__file__, "n/a", "n/a", fresh_sim_version="v3")), f, indent=2)
    print(f"  fresh={fd:+.6e}  retention={ret*100 if ret else 0:.1f}%  wall={wall:.0f}s", flush=True)

print("\nAll fresh-sims done.")
