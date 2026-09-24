#!/usr/bin/env python
"""3D landscape diagnostics + rule-prescribed masked greedy (post 2026-07-15).

Framing (2026-07-16): adjoint usability in the thresholded 3D combinatorial
landscape is a MEASURABLE property with spatial, depth and ordinal structure.
Diagnostics are rule-based (no hand-picked masks / K / phase) so the exact
rules frozen on anchor A can be applied blind on anchor B:

  Rule M (pollution mask) — statistics propose, probes dispose:
      suspicion zones = contiguous column runs within D_NBHD_NM of a
      source/monitor plane hosting >=1 pixel in the global |O| >= Q_TAIL
      tail; each zone's N_PROBE top-|gain| pixels get a single-flip
      incremental FD; zone is masked iff sign-match < N_PROBE/N_PROBE.
      (Pure |O| statistics CANNOT separate the clean left band from the
      polluted right band — verified offline 2026-07-16: both host tail
      pixels; the winning masked top-5 pixels (5,48),(6,47),(8,46) sit in
      the left near-source region.  Probe disposal is evidence-forced.)

  Rule K (ranking horizon) — K_eff = first rank whose measured marginal
      dF falls into the +/- Z_KEFF * sigma_single random band.  The greedy
      measures every nominated flip individually, so each round yields the
      rank curve for free.

  Rule P (phase) — default i_conj_a (physics + anchor-A adjudication);
      every round runs a zero-sim phase scan (top-K Jaccard stability
      plateau) and checks the variant's phase sits inside the plateau;
      a round with top-5 hit-rate < 3/5 escalates to an explicit
      4-variant flip-response adjudication.

Per-round first-class observables (NOT just dF): gain distribution,
rank curve (K_eff), suspicion zones + mask drift, phase plateau + arg(a),
accepted-pixel spatial distribution, predicted/measured compression ratio.

Tasks:
  diagnose   rule-based anchor diagnosis -> diagnosis_<label>.json
             (--replay consumes existing precision/linearity jsons, 0 sims)
  greedy     masked greedy rounds: jac (2 sims) + sequential verified flips
  randbase   incremental matched random baseline (one session, ~40s/draw)
  zones      offline: print suspicion zones from a saved jac
  phasescan  offline: print phase-scan plateau from a saved jac

Usage:
  python -m wdm3_3d.wdm3_3d_landscape --task diagnose --seed 7 --port 1 --replay
  python -m wdm3_3d.wdm3_3d_landscape --task randbase --seed 7 --port 1 --k 1 --draws 30
  python -m wdm3_3d.wdm3_3d_landscape --task greedy   --seed 7 --port 1 --rounds 5
"""

import json
import time
from pathlib import Path

import numpy as np

from .wdm3_3d_device import (
    DESIGN_X_NM,
    NX_2D,
    NY_2D,
    WL_NM,
    _build_3d_sim,
    _close_sim,
    load_warmstart_params,
)
from .wdm3_3d_audit import read_formal_target_3d
from .wdm3_3d_smoke import _apply_params_to_design
from .wdm3_3d_jac import (
    GRAD_VARIANTS,
    RESULTS_DIR,
    _flip_gain,
    _flip_single_pixel,
    _get_jac,
    _label,
    _now_iso,
    _save_json,
    _sweep_stray_engines,
)

LANDSCAPE_DIR = RESULTS_DIR.parent / "landscape"

# Source/monitor plane x (device.py: source at left edge - 100nm, P*_prof at
# right edge + 100nm; P_in_prof at left edge - 200nm is behind the source).
SRC_PLANE_X_NM = DESIGN_X_NM[0] - 100
MON_PLANE_X_NM = DESIGN_X_NM[1] + 100

# ── Rule parameters: FROZEN on anchor A (2026-07-16), applied blind on B ──
RULES = {
    "q_tail": 0.995,       # global |O| quantile defining the extreme tail
    "d_nbhd_nm": 500.0,    # near-field neighborhood of a source/monitor plane
    "bridge": 2,           # max column gap when merging tail runs into a zone
    "n_probe": 2,          # single-flip probes per suspicion zone
    "z_keff": 2.0,         # K_eff band: marginal dF <= z * sigma_single
    "sigma_single": 5.4e-4,  # random single-flip sigma (K=5 batch 1.2e-3/sqrt5)
    "jaccard_thr": 0.8,    # phase-plateau stability threshold
    "gain_band_frac": 0.9,  # phase must reach this fraction of max top-K gain
    "phase_scan_k": 20,    # top-K set size for the phase scan
    "phase_scan_n": 72,    # phi resolution (5 deg)
    "hit_escalate": 0.6,   # round top-5 hit-rate below this -> re-adjudicate phase
    "accept_floor": 0.0,   # greedy accepts a flip iff measured dF > this
}


def _col_of(idx):
    return int(idx) // NY_2D


def _col_centers_nm():
    dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
    return DESIGN_X_NM[0] + (np.arange(NX_2D) + 0.5) * dx


# ═══════════════════════════════════════════════════════════════════════════
# Offline diagnostics (0 sims)
# ═══════════════════════════════════════════════════════════════════════════

def propose_suspicion_zones(O_pix, rules=RULES):
    """Rule M stage 1: statistics propose suspicion zones.

    A zone is a contiguous run (gaps <= bridge merged) of columns that
    (a) lie within d_nbhd_nm of the source or monitor plane and
    (b) host >= 1 pixel of the global |O| >= q_tail tail.
    Runs whose end is within `bridge` columns of the design boundary are
    extended to the boundary.
    """
    absO = np.abs(np.asarray(O_pix)).reshape(NX_2D, NY_2D)
    thr = float(np.quantile(absO, rules["q_tail"]))
    tail_cols = np.unique(np.where(absO >= thr)[0])

    xc = _col_centers_nm()
    dist_src = np.abs(xc - SRC_PLANE_X_NM)
    dist_mon = np.abs(xc - MON_PLANE_X_NM)
    nbhd = np.minimum(dist_src, dist_mon) <= rules["d_nbhd_nm"]

    suspects = sorted(int(c) for c in tail_cols if nbhd[c])
    zones = []
    for c in suspects:
        if zones and c - zones[-1]["hi"] <= rules["bridge"] + 1:
            zones[-1]["hi"] = c
            zones[-1]["tail_cols"].append(c)
        else:
            zones.append({"lo": c, "hi": c, "tail_cols": [c]})
    for z in zones:
        if z["lo"] <= rules["bridge"]:
            z["lo"] = 0
        if NX_2D - 1 - z["hi"] <= rules["bridge"]:
            z["hi"] = NX_2D - 1
        z["side"] = ("src" if dist_src[(z["lo"] + z["hi"]) // 2]
                     <= dist_mon[(z["lo"] + z["hi"]) // 2] else "mon")
        z["n_tail"] = len(z["tail_cols"])
    return zones, thr


def zone_probe_plan(zones, gain, n_probe=RULES["n_probe"]):
    """Rule M stage 2 plan: each zone's n_probe top-|gain| pixels."""
    ix_of = np.arange(len(gain)) // NY_2D
    for z in zones:
        in_zone = (ix_of >= z["lo"]) & (ix_of <= z["hi"])
        cand = np.where(in_zone)[0]
        z["probe_idx"] = [int(i) for i in
                          cand[np.argsort(np.abs(gain[cand]))[::-1][:n_probe]]]
    return zones


def mask_from_zones(zones):
    """Column ranges of zones whose probe verdict failed."""
    return [(z["lo"], z["hi"]) for z in zones if z.get("masked")]


def apply_col_mask(n_dims, mask_ranges):
    """Boolean exclusion array from [(lo, hi), ...] column ranges."""
    excl = np.zeros(n_dims, dtype=bool)
    if mask_ranges:
        ix_of = np.arange(n_dims) // NY_2D
        for lo, hi in mask_ranges:
            excl |= (ix_of >= lo) & (ix_of <= hi)
    return excl


def variant_phi_deg(variant, a):
    """Absolute scan phase of each gradient variant: g = Re(e^{i phi} O)."""
    arg_a = np.degrees(np.angle(a))
    return {"raw": 0.0, "i": 90.0, "conj_a": (-arg_a) % 360,
            "i_conj_a": (90.0 - arg_a) % 360}[variant]


def phase_scan(O_pix, params, mask_ranges=None, rules=RULES):
    """Zero-sim phase scan: top-K flip-gain SUM S(phi) + set stability.

    The adjudicating observable is S(phi) = sum of the top-K flip gains —
    its peak marks the phase that promises the most improving flips
    (2026-07-15: peak 202.5 deg, i_conj_a at 222.2 adjacent, raw in the
    trough).  The gain band is the arc where S(phi) >= gain_band_frac *
    max S; adjacent-phi Jaccard is reported as a stability check, not as
    the verdict (Jaccard alone is too permissive — all 4 variants pass it
    on anchor A).
    """
    O = np.asarray(O_pix).flatten()
    params = np.asarray(params, dtype=float).reshape(-1)
    sgn = 1.0 - 2.0 * (params > 0.5)
    excl = apply_col_mask(len(O), mask_ranges)
    K = rules["phase_scan_k"]
    n = rules["phase_scan_n"]

    phis = np.arange(n) * (360.0 / n)
    sets, S = [], []
    for phi in phis:
        gain = np.real(np.exp(1j * np.deg2rad(phi)) * O) * sgn
        gain[excl] = -np.inf
        order = np.argsort(gain)[::-1][:K]
        sets.append(frozenset(order.tolist()))
        S.append(float(np.sum(gain[order])))
    S = np.array(S)

    jac_adj = []
    for i in range(n):
        s1, s2 = sets[i], sets[(i + 1) % n]
        jac_adj.append(len(s1 & s2) / len(s1 | s2))

    peak_i = int(np.argmax(S))
    in_band = S >= rules["gain_band_frac"] * S[peak_i]
    bands = _circular_runs(phis, in_band)

    return {"phi_step_deg": 360.0 / n,
            "S_topk": [float(s) for s in S],
            "S_peak_deg": float(phis[peak_i]),
            "S_peak": float(S[peak_i]),
            "gain_bands_deg": bands,
            "jaccard_adjacent": [float(j) for j in jac_adj],
            "jaccard_min": float(min(jac_adj)),
            "jaccard_mean": float(np.mean(jac_adj))}


def _circular_runs(phis, flags):
    """Contiguous True runs over a circular phi axis -> [[lo, hi], ...]."""
    n = len(phis)
    runs = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            runs.append([float(phis[i]), float(phis[(j + 1) % n])])
            i = j + 1
        else:
            i += 1
    if len(runs) >= 2 and flags[0] and flags[-1]:
        runs[0][0] = runs[-1][0]
        runs.pop()
    return runs


def phi_in_plateaus(phi, plateaus):
    for lo, hi in plateaus:
        if lo <= hi:
            if lo <= phi <= hi:
                return True
        elif phi >= lo or phi <= hi:  # wrapped arc
            return True
    return False


def k_eff_from_rank_curve(dF_seq, rules=RULES):
    """Rule K: horizon ends at the first marginal inside the random band."""
    band = rules["z_keff"] * rules["sigma_single"]
    for i, dF in enumerate(dF_seq):
        if dF <= band:
            return i
    return len(dF_seq)


def gain_stats(gain, mask_ranges=None):
    """Distribution observables of the (masked) flip-gain field."""
    excl = apply_col_mask(len(gain), mask_ranges)
    g = gain[~excl]
    top20 = np.argsort(np.where(excl, -np.inf, gain))[::-1][:20]
    return {
        "n_pool": int(np.sum(~excl)),
        "n_positive": int(np.sum(g > 0)),
        "positive_fraction": float(np.mean(g > 0)),
        "quantiles": {q: float(np.quantile(g, float(q)))
                      for q in ("0.5", "0.9", "0.99", "0.999")},
        "max": float(np.max(g)),
        "top20_cols": sorted(set(_col_of(i) for i in top20)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Incremental evaluation session
# ═══════════════════════════════════════════════════════════════════════════

class IncrementalSession:
    """One CAD session; pixel flips as incremental rect add/delete + undo.

    28-40s per evaluation vs 5.5min fresh-sim; cross-session baseline
    verified bit-identical on 2026-07-15 (0.041841).
    """

    def __init__(self, params, target_port, wl_nm=WL_NM, tag="incr3d"):
        self.target_port = target_port
        self.state = np.asarray(params, dtype=float).reshape(-1) > 0.5
        self.sim = _build_3d_sim(tag=tag)
        _apply_params_to_design(self.sim.fdtd,
                                np.asarray(params, dtype=float).reshape(-1))
        self.fom = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        _close_sim(self.sim)
        _sweep_stray_engines()

    def _run_fom(self):
        self.sim.fdtd.run()
        return read_formal_target_3d(self.sim.fdtd, self.target_port)["abs2"]

    def run_baseline(self):
        self.fom = self._run_fom()
        return self.fom

    def _toggle(self, idx):
        self.sim.fdtd.switchtolayout()
        _flip_single_pixel(self.sim.fdtd, int(idx), to_si=not self.state[idx])
        self.state[idx] = not self.state[idx]

    def measure(self, idx_list, keep=False):
        """Flip pixels, run, return |a|^2; revert unless *keep*."""
        for i in idx_list:
            self._toggle(i)
        f = self._run_fom()
        if keep:
            self.fom = f
        else:
            for i in idx_list:
                self._toggle(i)
        return f

    def try_flip(self, idx, accept_floor=0.0):
        """Toggle one pixel, run, keep iff dF > accept_floor.

        Rejection reverts the geometry WITHOUT re-running (the next
        measurement runs anyway) — one solver run per candidate either way.
        Returns (dF, accepted).
        """
        self._toggle(idx)
        f = self._run_fom()
        dF = f - self.fom
        if dF > accept_floor:
            self.fom = f
            return dF, True
        self._toggle(idx)
        return dF, False

    def params(self):
        return self.state.astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# Task: diagnose — rule-based anchor diagnosis (Rule M + Rule P)
# ═══════════════════════════════════════════════════════════════════════════

def _replay_measurements(label, out_dir):
    """idx -> measured single-flip dF from existing precision/linearity jsons."""
    meas = {}
    for pat in (f"precision_{label}_*.json",):
        for p in sorted(Path(out_dir).glob(pat)):
            data = json.load(open(p))
            for e in data.get("entries", []):
                meas[int(e["idx"])] = {"dF": float(e["delta_fom"]),
                                       "source": p.name}
    return meas


def diagnose_anchor(seed, target_port, wl_nm=WL_NM, variant="i_conj_a",
                    replay=False, rules=RULES, out_dir=None, jac_dir=None):
    """Rule-based diagnosis: suspicion zones -> probe verdicts -> mask;
    phase scan sanity.  Saves diagnosis_<label>.json (the prescription that
    the greedy — and anchor B — consume without human editing).
    """
    out_dir = Path(out_dir) if out_dir else LANDSCAPE_DIR
    jac_dir = Path(jac_dir) if jac_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    print(f"\n{'='*60}")
    print(f"DIAGNOSE: {label}  variant={variant}  mode={'replay' if replay else 'live'}")
    print(f"{'='*60}")

    jac = _get_jac(seed, target_port, wl_nm, jac_dir, reuse=True)
    O_pix = jac["O_pix"]
    a = jac["a"]
    params = np.asarray(jac["params"], dtype=float).reshape(-1)

    zones, tail_thr = propose_suspicion_zones(O_pix, rules)

    # ── Blind variant adjudication (variant="auto") ──
    # Provisional CONSERVATIVE mask = every suspicion zone masked; the
    # variant whose phase sits inside the S(phi) gain band wins, tie-broken
    # by circular distance to the peak.  No human in the loop (seed10 blind
    # protocol 2026-07-16: transfer the RULE, never seed7's variant).
    variant_selection = None
    if variant == "auto":
        prov_mask = [(z["lo"], z["hi"]) for z in zones]
        prov_scan = phase_scan(O_pix, params, prov_mask, rules)

        def _peak_dist(v):
            d = (variant_phi_deg(v, a) - prov_scan["S_peak_deg"] + 180) % 360
            return abs(d - 180)

        in_band = [v for v in GRAD_VARIANTS
                   if phi_in_plateaus(variant_phi_deg(v, a),
                                      prov_scan["gain_bands_deg"])]
        pool = in_band if in_band else list(GRAD_VARIANTS)
        variant = min(pool, key=_peak_dist)
        variant_selection = {
            "mode": "auto",
            "provisional_mask": [list(m) for m in prov_mask],
            "provisional_S_peak_deg": prov_scan["S_peak_deg"],
            "provisional_gain_bands_deg": prov_scan["gain_bands_deg"],
            "variants_in_band": in_band,
            "selected": variant,
            "flag": None if in_band else "NO variant in provisional gain band"
                                         " - nearest-to-peak fallback",
        }
        print(f"  [auto] S peak {prov_scan['S_peak_deg']:.1f} deg, "
              f"in-band variants: {in_band or 'NONE'} -> selected {variant}")

    gain = _flip_gain(jac["grads"][variant], params)
    zones = zone_probe_plan(zones, gain, rules["n_probe"])
    print(f"  |O| tail threshold (q={rules['q_tail']}): {tail_thr:.3e}")
    for z in zones:
        print(f"  suspicion zone ix=[{z['lo']},{z['hi']}] side={z['side']} "
              f"tail_cols={z['tail_cols']} probes={z['probe_idx']}")

    # ── Probe disposal ──
    t0 = time.time()
    if replay:
        meas = _replay_measurements(label, jac_dir)
        session = None
    else:
        session = IncrementalSession(params, target_port, wl_nm,
                                     tag=f"diag3d_{label}")
    try:
        if session is not None:
            f0 = session.run_baseline()
            print(f"  baseline |a|^2 = {f0:.6f}")
        for z in zones:
            z["verdicts"] = []
            for idx in z["probe_idx"]:
                if replay:
                    m = meas.get(idx)
                    if m is None:
                        z["verdicts"].append({"idx": idx, "dF": None,
                                              "sign_match": None,
                                              "source": "unmeasured"})
                        continue
                    dF, src = m["dF"], m["source"]
                else:
                    f = session.measure([idx])
                    dF, src = f - session.fom, "live"
                match = bool((dF > 0) == (gain[idx] > 0))
                z["verdicts"].append({"idx": idx, "col": _col_of(idx),
                                      "gain_pred": float(gain[idx]),
                                      "dF": float(dF), "sign_match": match,
                                      "source": src})
                print(f"    probe idx={idx} (ix={_col_of(idx)}) "
                      f"pred={gain[idx]:+.4f} dF={dF:+.3e} "
                      f"{'MATCH' if match else 'MISS'} [{src}]")
            judged = [v for v in z["verdicts"] if v["sign_match"] is not None]
            z["masked"] = (len(judged) == 0
                           or any(not v["sign_match"] for v in judged))
            print(f"  zone ix=[{z['lo']},{z['hi']}]: "
                  f"{'MASKED' if z['masked'] else 'clean'}")
    finally:
        if session is not None:
            session.close()

    mask = mask_from_zones(zones)
    scan = phase_scan(O_pix, params, mask, rules)
    phis = {v: variant_phi_deg(v, a) for v in GRAD_VARIANTS}
    in_band = phi_in_plateaus(phis[variant], scan["gain_bands_deg"])
    print(f"  mask = {mask}")
    print(f"  S(phi) peak {scan['S_peak_deg']:.1f} deg, gain bands (deg) = "
          f"{scan['gain_bands_deg']}  {variant} phi={phis[variant]:.1f}  "
          f"in_band={in_band}")

    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "variant": variant,
        "variant_selection": variant_selection,
        "timestamp": _now_iso(),
        "mode": "replay" if replay else "live",
        "rules": rules,
        "arg_a_deg": float(np.degrees(np.angle(a))),
        "tail_threshold": tail_thr,
        "zones": zones,
        "mask": [list(m) for m in mask],
        "phase_scan": scan,
        "variant_phi_deg": phis,
        "variant_in_gain_band": bool(in_band),
        "gain_stats_masked": gain_stats(gain, mask),
        "wall_s": time.time() - t0,
    }
    _save_json(result, out_dir / f"diagnosis_{label}.json")
    return result


def load_diagnosis(seed, target_port, wl_nm=WL_NM, out_dir=None):
    out_dir = Path(out_dir) if out_dir else LANDSCAPE_DIR
    p = out_dir / f"diagnosis_{_label(seed, target_port, wl_nm)}.json"
    if not p.exists():
        return None
    return json.load(open(p))


# ═══════════════════════════════════════════════════════════════════════════
# Task: greedy — adjoint proposes, FDTD disposes (per-flip verification)
# ═══════════════════════════════════════════════════════════════════════════

def greedy_run(seed, target_port, wl_nm=WL_NM, rounds=5, k_max=10,
               variant="i_conj_a", rules=RULES, out_dir=None, jac_dir=None,
               stop_sigma=1.2e-3, stop_patience=2):
    """Masked greedy: per round 1 jac (2 sims) + sequential verified flips.

    Every nominated flip is measured individually (incremental, ~40s) in
    rank order on top of the accepted structure — acceptance IS the
    verification, and the measured rank curve is the K_eff observable.
    Mask is static from the anchor diagnosis; suspicion zones are re-proposed
    offline every round and drift is flagged, not acted on (mask boundary
    scan stays a triggered follow-up, per 2026-07-15 decision).
    """
    out_dir = Path(out_dir) if out_dir else LANDSCAPE_DIR
    jac_dir = Path(jac_dir) if jac_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag = load_diagnosis(seed, target_port, wl_nm, out_dir)
    if diag is None:
        raise RuntimeError("no diagnosis found — run --task diagnose first "
                           "(diagnosis prescribes; greedy only executes)")
    mask = [tuple(m) for m in diag["mask"]]

    # ── Resume from the last completed round ──
    done = sorted(out_dir.glob(f"greedy_{label}_r*.json"))
    if done:
        last = json.load(open(done[-1]))
        params = np.asarray(np.load(out_dir / last["params_npz"])["params"],
                            dtype=float)
        start_round = last["round"] + 1
        history = last["history"]
        print(f"  [resume] round {start_round}, cum_dF={history[-1]:+.4e}")
    else:
        params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
        start_round = 0
        history = []

    print(f"\n{'='*60}")
    print(f"GREEDY: {label}  variant={variant}  mask={mask}  "
          f"rounds {start_round}..{rounds - 1}  k_max={k_max}")
    print(f"{'='*60}")

    for r in range(start_round, rounds):
        t0 = time.time()
        print(f"\n--- round {r} ---")
        jac = _get_jac(seed, target_port, wl_nm, jac_dir, reuse=True,
                       params=params, label_suffix=f"_g{r}")
        O_pix, a = jac["O_pix"], jac["a"]
        base = jac["baseline"]["abs2"]
        gain = _flip_gain(jac["grads"][variant], params)

        # offline per-round diagnostics (0 sims)
        zones, _ = propose_suspicion_zones(O_pix, rules)
        scan = phase_scan(O_pix, params, mask, rules)
        phi = variant_phi_deg(variant, a)
        in_band = phi_in_plateaus(phi, scan["gain_bands_deg"])
        if not in_band:
            print(f"  [WARN] variant phi={phi:.1f} outside gain bands "
                  f"{scan['gain_bands_deg']} — phase drift alarm")

        excl = apply_col_mask(len(params), mask)
        sel = np.where(excl, -np.inf, gain)
        nom_idx = [int(i) for i in np.argsort(sel)[::-1][:k_max]
                   if sel[int(i)] > 0]

        nominations = []
        with IncrementalSession(params, target_port, wl_nm,
                                tag=f"greedy3d_{label}_r{r}") as S:
            f0 = S.run_baseline()
            drift = abs(f0 - base)
            print(f"  baseline |a|^2 = {f0:.6f} (jac session {base:.6f}, "
                  f"drift {drift:.1e})")
            for rank, idx in enumerate(nom_idx, start=1):
                t_e = time.time()
                dF, accept = S.try_flip(idx, rules["accept_floor"])
                nominations.append({
                    "rank": rank, "idx": idx,
                    "ix": _col_of(idx), "iy": int(idx) % NY_2D,
                    "gain_pred": float(gain[idx]),
                    "dF_meas": float(dF),
                    "accepted": bool(accept),
                    "wall_s": time.time() - t_e,
                })
                print(f"  [{rank}/{len(nom_idx)}] idx={idx} (ix={_col_of(idx)}) "
                      f"pred={gain[idx]:+.4f} dF={dF:+.4e} "
                      f"{'ACCEPT' if accept else 'reject'} "
                      f"({nominations[-1]['wall_s']:.0f}s)", flush=True)
            params = S.params()
            f_end = S.fom

        accepted = [n for n in nominations if n["accepted"]]
        net = f_end - f0
        history.append(float(net))
        k_eff = k_eff_from_rank_curve([n["dF_meas"] for n in nominations], rules)
        hit5 = (sum(1 for n in nominations[:5] if n["dF_meas"] > 0) / 5.0
                if len(nominations) >= 5 else float("nan"))
        pred_sum = sum(n["gain_pred"] for n in accepted)
        compression = (sum(n["dF_meas"] for n in accepted) / pred_sum
                       if pred_sum else float("nan"))
        edge_flips = [n["ix"] for n in accepted
                      if any(abs(n["ix"] - b) <= 2 for mr in mask for b in mr)]

        params_npz = f"greedy_{label}_r{r}_params.npz"
        np.savez(out_dir / params_npz, params=params)
        record = {
            "label": label, "round": r, "variant": variant,
            "timestamp": _now_iso(),
            "mask": [list(m) for m in mask],
            "baseline_abs2": float(f0), "end_abs2": float(f_end),
            "net_dF": float(net), "history": history,
            "arg_a_deg": float(np.degrees(np.angle(a))),
            "phase": {"variant_phi_deg": phi, "in_gain_band": bool(in_band),
                      "S_peak_deg": scan["S_peak_deg"],
                      "gain_bands_deg": scan["gain_bands_deg"],
                      "jaccard_min": scan["jaccard_min"]},
            "suspicion_zones": [{k: z[k] for k in
                                 ("lo", "hi", "side", "tail_cols", "n_tail")}
                                for z in zones],
            "gain_stats_masked": gain_stats(gain, mask),
            "nominations": nominations,
            "k_eff": k_eff,
            "hit_rate_top5": hit5,
            "phase_escalation": bool(hit5 < rules["hit_escalate"]),
            "compression_meas_over_pred": float(compression),
            "accepted_ix": [n["ix"] for n in accepted],
            "mask_edge_flips": edge_flips,
            "params_npz": params_npz,
            "wall_s": time.time() - t0,
        }
        _save_json(record, out_dir / f"greedy_{label}_r{r}.json")
        print(f"  round {r}: net dF={net:+.4e}  accepted {len(accepted)}"
              f"/{len(nom_idx)}  K_eff={k_eff}  hit@5={hit5:.1f}  "
              f"compression={compression:.4f}")
        if edge_flips:
            print(f"  [FLAG] accepted flips near mask edge: ix={edge_flips} "
                  f"-> consider mask boundary scan")
        if record["phase_escalation"]:
            print("  [FLAG] top-5 hit-rate below threshold -> explicit "
                  "4-variant flip-response adjudication required before "
                  "next round")
            break
        if (len(history) >= stop_patience
                and all(h < stop_sigma for h in history[-stop_patience:])):
            print(f"  [STOP] {stop_patience} consecutive rounds with net dF "
                  f"< {stop_sigma:.1e}")
            break

    print(f"\n  cumulative dF = {sum(history):+.4e} over {len(history)} rounds")
    return history


# ═══════════════════════════════════════════════════════════════════════════
# Task: randbase — incremental matched random baseline
# ═══════════════════════════════════════════════════════════════════════════

def randbase_incremental(seed, target_port, wl_nm=WL_NM, k=5, draws=20,
                         rng_seed=5678, use_mask=True, out_dir=None,
                         jac_dir=None):
    """Matched random flips in ONE incremental session (~40s/draw).

    k=1 with draws>=20 estimates sigma_single directly (Rule K input);
    k=5 is the push-matched null.  Pool excludes the diagnosed mask so
    random draws come from the same interior region as the nominations.
    """
    out_dir = Path(out_dir) if out_dir else LANDSCAPE_DIR
    jac_dir = Path(jac_dir) if jac_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, jac_dir, reuse=True)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    mask = []
    if use_mask:
        diag = load_diagnosis(seed, target_port, wl_nm, out_dir)
        if diag is None:
            raise RuntimeError("no diagnosis found — run --task diagnose "
                               "first, or pass --no-mask")
        mask = [tuple(m) for m in diag["mask"]]
    pool = np.where(~apply_col_mask(len(params), mask))[0]
    rng = np.random.RandomState(rng_seed)

    print(f"\n{'='*60}")
    print(f"RANDBASE (incremental): {label}  k={k}  draws={draws}  "
          f"mask={mask}  pool={len(pool)}")
    print(f"{'='*60}")

    t0 = time.time()
    entries = []
    with IncrementalSession(params, target_port, wl_nm,
                            tag=f"randinc3d_{label}") as S:
        f0 = S.run_baseline()
        print(f"  baseline |a|^2 = {f0:.6f}")
        for d_i in range(draws):
            flip_idx = rng.choice(pool, size=int(k), replace=False)
            t_e = time.time()
            f = S.measure(list(flip_idx), keep=False)
            dF = f - f0
            entries.append({"draw": d_i,
                            "flip_idx": [int(i) for i in flip_idx],
                            "delta_fom": float(dF)})
            print(f"  [{d_i + 1}/{draws}] dF={dF:+.4e} "
                  f"({time.time() - t_e:.0f}s)", flush=True)

    deltas = np.array([e["delta_fom"] for e in entries])
    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "K": int(k), "draws": draws, "rng_seed": rng_seed,
        "mask": [list(m) for m in mask], "pool": int(len(pool)),
        "timestamp": _now_iso(),
        "baseline_abs2": float(f0),
        "entries": entries,
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)),
        "delta_max": float(deltas.max()),
        "sigma_single_implied": float(deltas.std(ddof=1) / np.sqrt(k)),
        "wall_s": time.time() - t0,
    }
    mask_tag = "_mask" if mask else ""
    _save_json(result, out_dir / f"randinc_{label}_k{k}_d{draws}{mask_tag}.json")
    print(f"\n  dF: mean={result['delta_mean']:+.4e} "
          f"std={result['delta_std']:.3e} max={result['delta_max']:+.4e}  "
          f"sigma_single~{result['sigma_single_implied']:.3e}")
    print(f"  wall: {result['wall_s']:.0f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    import lum_backend  # noqa: F401 — verifies backend path
    import _lumopt_compat  # noqa: F401

    parser = argparse.ArgumentParser(
        description="3D landscape diagnostics + rule-prescribed masked greedy",
    )
    parser.add_argument("--task", required=True,
                        choices=["diagnose", "greedy", "randbase", "zones",
                                 "phasescan"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--wl", type=float, default=1550.0)
    parser.add_argument("--variant", default="i_conj_a",
                        choices=list(GRAD_VARIANTS) + ["auto"])
    parser.add_argument("--sigma-single", type=float, default=None,
                        help="local random single-flip sigma (Rule K band); "
                             "MUST be measured at the anchor under test")
    parser.add_argument("--stop-sigma", type=float, default=None,
                        help="greedy early-stop threshold (local random "
                             "5-flip std); defaults to seed7's 1.2e-3")
    parser.add_argument("--replay", action="store_true",
                        help="diagnose: consume existing probe jsons (0 sims)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=5678)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    rules = dict(RULES)
    if args.sigma_single is not None:
        rules["sigma_single"] = args.sigma_single

    if args.task == "diagnose":
        diagnose_anchor(args.seed, args.port, args.wl, variant=args.variant,
                        replay=args.replay, rules=rules, out_dir=out_dir)
    elif args.task == "greedy":
        greedy_run(args.seed, args.port, args.wl, rounds=args.rounds,
                   k_max=args.k_max, variant=args.variant, rules=rules,
                   stop_sigma=(args.stop_sigma if args.stop_sigma is not None
                               else 1.2e-3),
                   out_dir=out_dir)
    elif args.task == "randbase":
        randbase_incremental(args.seed, args.port, args.wl, k=args.k,
                             draws=args.draws, rng_seed=args.rng_seed,
                             use_mask=not args.no_mask, out_dir=out_dir)
    elif args.task in ("zones", "phasescan"):
        jac = _get_jac(args.seed, args.port, args.wl, None, reuse=True)
        params = np.asarray(jac["params"], dtype=float).reshape(-1)
        if args.task == "zones":
            zones, thr = propose_suspicion_zones(jac["O_pix"])
            gain = _flip_gain(jac["grads"][args.variant], params)
            zones = zone_probe_plan(zones, gain)
            print(f"tail threshold: {thr:.3e}")
            for z in zones:
                print(z)
        else:
            scan = phase_scan(jac["O_pix"], params)
            print(f"S(phi) peak: {scan['S_peak_deg']:.1f} deg  "
                  f"(S={scan['S_peak']:.3e})")
            print(f"gain bands (deg): {scan['gain_bands_deg']}")
            print(f"jaccard: min={scan['jaccard_min']:.3f} "
                  f"mean={scan['jaccard_mean']:.3f}")
            for v in GRAD_VARIANTS:
                phi = variant_phi_deg(v, jac["a"])
                print(f"  {v}: phi={phi:.1f} in_gain_band="
                      f"{phi_in_plateaus(phi, scan['gain_bands_deg'])}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
