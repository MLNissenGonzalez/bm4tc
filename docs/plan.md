# Plan: what is left

The `ousterhout` refactor (2026-10-06 to 2026-10-07) simplified bm4tc before the
experiments are re-run for the journal version of the rejected NeurIPS submission
27079. Phases 0–7 are done: seam tests and benchmark, deletions, one metrics module,
one Trainer, model-owned caches, studies and `run.json`, the `bm4tc/{core,analysis,
pipeline}` package with the CLI, JEM in the pipeline, figures from paper manifests,
and these docs. How each was decided is in [decisions.md](decisions.md) (D1–D77); the
per-phase plan is in the git history (`plan.md` before this file).

## Phase 8: reproduce the journal paper

Run the studies on the lab HPC, then `python -m bm4tc figures paper`.

Order (warm studies need their `warm_from` study trained):

1. Pilots that set defaults (one-off, by hand):
   - PGD-5 vs PGD-10 AT training on `mnist12` d3r40 α=0 (D43): if PGD-5 loses more
     than ~1 point of robust accuracy at PGD-40 evaluation, keep 10; otherwise set
     `num_steps: 5` in `configs/trainer/at.yaml`.
   - The norm-control target for cold MNIST runs, 0 vs n·ln d / 2 (D59).
2. `spirals_nat`, `spirals_capacity`, `spirals_embedding`, `mnist12_nat`,
   `mnist_capacity`, `jem_mnist12_nat` (independent; cold).
3. `spirals_at`, `mnist12_at`, `jem_mnist12_at` (warm).
4. Set `arch` in `mnist_nat` / `mnist_at` from `mnist_capacity` (D44's two
   criteria), write `jem_mnist_{nat,at}` (copies of the `jem_mnist12_*` studies on
   `mnist`, `arch` = the chosen MPS), then run `mnist_nat`, `jem_mnist_nat`,
   `mnist_at`, `jem_mnist_at`.
5. `python -m bm4tc figures paper`; commit `figures/journal/`.

Still to provide for Phase 8:
- A `run paper`-style launcher across studies (the executor already runs a DAG;
  today one launch runs one study, D66).
- Compute estimates per study (HPO budget × cells × seeds).
- Gibbs/SGLD purification is off by default; enable `analysis.sweep_purify` in the
  studies whose figures need it (`paper.yaml`'s `mnist_sweep_purify`).

## Phase 9 (separate track): tensorkrowch

Verify each upstream candidate in [decisions.md](decisions.md#tensorkrowch-upstream-candidates-tensorkrowch-116)
against the current tensorkrowch source; prototype as patches in a fork, run the
bm4tc suite against the fork, then propose upstream. Changes to the bm4tc numerical
core (`ConditionalBornMachine` contractions, `log_partition_function`, accumulate,
sampling) happen here and nowhere else (D30).

## Open items

- **Time series** (D6): datasets, grids and studies (NAT and AT; JEM if wanted).
- **Study settings** were copied from the old sweeps (epochs, patience, HPO spaces);
  review before Phase 8.
- **Trainer unification** for MPS and JEM: noted, not decided (see the note under
  the decisions table).
- **Adaptive attacks** a journal reviewer will ask for (rebuttal): BPDA/EOT through
  Gibbs purification, unrolled PGD through gradient purification. Not implemented.
- **Agent notes on the other laptop:** `CLAUDE.md`, `DEFERRED.md` and `.claude/`
  there may hold the old plan and issue list; move what still matters into GitHub
  issues or this file.
- `tests/unit/test_privacy.py::test_membership_inference_runs_end_to_end` fails when
  only `tests/unit tests/integration` run (order-dependent on the session `cbm`
  fixture); passes in the full suite.
