# Data and code for: measuring action-landscape acceptance loss and probe resolution

Reproduction package for

> *Measuring action-landscape acceptance loss and probe resolution in
> FDTD-driven binary inverse design of a silicon WDM multiplexer*

**Every table the paper cites is in the paper.** There is no supplementary file
and no appendix, so nothing in the manuscript defers to this package for a
number; the package carries the underlying records instead.

## Contents

| Path | What it is |
|---|---|
| `data_tables/body/` | Machine-readable re-emission of the 8 main-text table floats, parsed from the manuscript source so they cannot drift from the paper; `contrast_a.csv` and `contrast_b.csv` are the two panels of one float, so 9 files back those 8 |
| `data_tables/gate_loss_provenance.csv` | Per-anchor assignment of each gate-loss value to the one trajectory it was read from |
| `data_tables/arm_inventory.csv` and `arm_inventory_summary.json` | Measured policy-cell inventory and its summary counts for the arm-level low-gain audit |
| `data_tables/null_control_strata.json` + `null_control_records/` | The two-strata separation test behind the matched-control sentence, and the raw records it is computed from |
| `data_tables/probe_pool_size_audit.json` | Probe-pool composition audit behind the single-size statement: the modal patch size, its median share of positive probe mass, and the shift in the frozen loss when the pool is restricted to it |
| `data_tables/floor_identity_audit.json` | Action-level check of the overlap identity: every recorded verdict against the threshold the rule applied, and the probe steps whose applied floor is identically zero |
| `data_tables/p0_predictor_accuracy.json` | Phase-0 predictor accuracies on the harmonized cohort, with the earlier twelve-anchor snapshot reproduced as a self-check |
| `probe_records/` | 71 probe trajectories; every evaluated `dJ` is recorded whether or not it was accepted, so the raw data are verifier-agnostic |
| `fresh_records/` | 50 independent fresh-simulation results used as the endpoint |
| `floor_sweep_records/` | The paired floor-sweep arms behind the cost/loss numbers in the Discussion: per anchor, `k=0` keeps the probe's floor and `k=1` removes it, each arm with its own fresh simulation |
| `panel_records/` | Per anchor, the frozen fixed-action panel (the candidate actions) and the `S_0` baseline measurement that every checkpoint delta is differenced against |
| `probe_readout_audit/` | How the per-action readout settles (same panel in frozen and shuffled order), a replay of the probe's own twenty draws at two anchors, and the settled-protocol counterfactual; `panels/` holds the two panels those Step-0 records are read against |
| `route_cost_records/` | The seed23 boundary comparison: both routed arms with their fresh simulations, and the readout of the criterion that was fixed before it ran |
| `protocol/` | The frozen protocol every record was produced under, including the gate order, the rule definitions, and the measured quantity |
| `scripts/` | Analysis and figure scripts, including the auditors used to check the manuscript |
| `lumerical/` | Simulator-side sources: the FDTD builders, the orchestration and backend code, and the warm-start parameter vectors |

## Labels inside the records

The JSON records under `fresh_records/` and the pipeline sources under
`scripts/` and `lumerical/` are shipped exactly as they were produced, so they
still carry the metadata and paths of the pipeline that wrote them, including
the internal name of the run directory each record came from. That name is the
pipeline's own label for the campaign generation. The paper and
`data_tables/arm_inventory.csv` use public labels instead: `current` for the
campaign the paper reports, and `earlier` for an older generation that is used
only as a fallback. The records are not rewritten to remove the internal label,
because they are the raw artefacts the paper's numbers come from.

## Reconstructing a committed geometry

`fresh_records/` reports the endpoint only (`fresh_dJ`, `retention`,
`n_accepts`); the committed pixel pattern is not stored beside it, because every
record here was produced inside a single process. It is nevertheless recoverable
from the shipped trajectories: a record's committed geometry is the binarised
warm-start vector with the pixels of every **accepted** step toggled once. An
accepted step is identified by the rule the analysis code uses,

    is_accept(t) = ("accept" in t) or t.endswith("_a")

which covers every arm in this package. The pixels of a step are in `indices`
when that key is present and in `idx` otherwise. Step **names** differ between
arm families --- `warmup_accept` / `static_accept` in the floor and static arms
against `wu_a` / `probe_a` / `bb_a` / `bs_a` / `sb_a` in the banded arms --- so
key on the rule above, not on the names. The replay scripts under `scripts/` are
family-specific by construction and each hard-codes one vocabulary; the rule
above is the general form.

## Two conventions worth knowing before reading the tables

1. **`online` is not `fresh`.** Online gain is what the trajectory accumulated
   under its own acceptance rule; fresh gain is an independent simulation of the
   device. Every headline number in the paper is a fresh value.
2. **The acceptance rule also governs the warm-up.** A comparison that changes
   the rule therefore changes the starting state as well, so cells labelled as
   rule contrasts are protocol-level contrasts, not isolated execution-stage
   verifier effects.

## Two seed16 batches, deliberately different

`fresh_records/seed16/floor_k*` and `floor_sweep_records/seed16/floor_k*` are
two runs of the same two arms, and they do not agree on the `k=0` arm. They are
both shipped, and they must not be mixed:

| Where | Which run | `k=0` fresh | `k=1` fresh | cost |
|---|---|---|---|---|
| `fresh_records/seed16/floor_k*` | published protocol | `0.1505297` | `0.0588800251` | `0.6088` |
| `floor_sweep_records/seed16/floor_k*` | the paired re-run quoted in the Discussion | `0.1566612` | `0.0588800251` | `0.624157` |

The `k=1` arms agree bit-for-bit. The `k=0` arms differ by 4.1%, because a long
acceptance chain is not bit-reproducible while an endpoint read is: the `k=0`
arm accumulates 53 accepted steps against 5 for `k=1`, and the committed
geometry is reached through a different assembly history. The Discussion quotes
the re-run pair, because that pair was measured as one batch.

Two definitions used in that paragraph, restated here so the numbers can be
recomputed from the records:

    cost(anchor) = 1 - fresh(k=1) / fresh(k=0)      (one paired run, per anchor)
    loss(anchor) = the probe's refused share of positive mass, i.e. the
                   `gate_loss_mass` column of data_tables/gate_loss_provenance.csv

and the quoted range is the ratio `cost / loss` over the four anchors.

## The recorded readout depends on evaluation position

`probe_readout_audit/` records what happens when the same action is read more
than once in one session. In `step0a` (the frozen panel order) and `step0b` (the
same panel, shuffled) the first pass differs from the second, the second and
third are bit-identical, and the size of the first-pass difference falls as the
action's position in the session advances: at seed23, Spearman correlation with
position is `+0.847` (frozen) and `+0.827` (shuffled), while correlating the same
action across the two orders gives `+0.361` (not significant). The offset
therefore tracks the position of the evaluation in the session, not the action.

`replicate.json` replays the probe's own twenty draws exactly as published and
reproduces every recorded `dJ` bit for bit at both anchors. `settled.json` reads
each draw twice and decides on the second read: the second read differs from the
first at 19 of 20 draws (seed23) and 20 of 20 (seed22), yet no draw crosses its
`T_null`, so no recorded label changes side. The gate loss under the settled rule
moves from `0.1925` to `0.2478` at seed23 and stays `0` at the clean anchor
seed22. The recorded labels at these two anchors are therefore unaffected, and
the dependence is on evaluation position within a session.

## Not included

Raw Lumerical project files are not redistributed, because of size and licence
terms. `lumerical/CANONICAL_PROJECT.md` records how the canonical pipeline is
built from source, and `lumerical/RUNBOOK.md` records a clean-machine
smoke-test.
