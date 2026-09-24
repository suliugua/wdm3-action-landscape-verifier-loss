# Runbook

## Clean-machine smoke test

Run these in order. Each step is cheap and fails loudly.

1. **Simulator reachable.** Launch the 3D FDTD engine once by hand and confirm
   the API path the scripts resolve matches the version in
   `CANONICAL_PROJECT.md`. The run log prints the resolved path at start-up.
2. **Device builds.** Build the WDM3 geometry from source with no monitors
   armed, and confirm the port list comes back as P1, P2, P3.
3. **One evaluation.** Run a single candidate flip end to end and confirm a
   finite `dJ` is written. A `dJ` of exactly zero on a first flip means the
   result schema was not read back, not that the flip was neutral.
4. **Probe instrument.** Run the 20-evaluation probe on one anchor from
   `probe_records/` and confirm `n_pos` and `n_accept` reproduce the values in
   `data_tables/body/gateaudit.csv` for that anchor.
5. **Fresh simulation.** Re-run the fresh-simulation tool on one anchor and
   confirm it reproduces the recorded `fresh_gain` in
   `data_tables/body/contrast_b.csv` to the recorded precision.

## Traps that cost time

* Check that writes land in an output path outside the package before any test
  run.
* The engine's cold start is paid on every reopen; a smoke test that opens the
  engine per evaluation measures the wrong thing.
* Kill the engine process tree explicitly when a run is abandoned, or the next
  run attaches to a stale session.
