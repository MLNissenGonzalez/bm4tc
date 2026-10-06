# Hand-off: `ousterhout` branch (2026-10-06)

For the next Claude session (on this laptop or the other one). Read this first, then
`plan.md`. Delete this file once its contents live in `AGENTS.md` (Phase 7).

## What this branch is

Martin is simplifying bm4tc (MPS Born machines for trustworthy classification) along
Ousterhout's *A Philosophy of Software Design*, before re-running every experiment for a
**journal version** of a NeurIPS submission (27079) that was rejected. Nothing in the code
has changed yet; so far the branch holds only documents:

| File | What | Status |
|---|---|---|
| `ousterhout.md` | compact summary of the book + general implications | done |
| `status.md` | survey of the codebase (§0–§10); **§11 = decisions D1–D46**; §12 = tensorkrowch upstream candidates | done; §11 is the source of truth for decisions |
| `plan.md` | target shape (§1), phases 0–9 (§2), untouched core (§3), sizes (§4), risks (§5), open questions (§6) | draft, Martin-approved apart from the open items below |
| `_paper/` (untracked, keep it untracked) | TPM 2026 paper (`main.tex`, outdated), NeurIPS rebuttal drafts | reference only |

`main` is untouched. Nothing has been pushed; ask before pushing.

## Done so far

- **Phase 0** (`92fb5b3b`): tag `pre-ousterhout` on `main` (local, not pushed);
  `pytest.ini`; seam test `tests/e2e/test_pipeline.py` (CPU, ~27 s, deterministic;
  `detection` is expected to change with D2); benchmark
  `python -m tests.bench.bench_train_step` (RTX 2080, d3r20 c64, batch 256:
  NAT 175 ms/step, AT PGD-10 1163 ms/step), the D31 baseline.
- **Phase 1** (`624b31e3`, `19d34ff3`): deleted `hpo.py`, the migration tools,
  `run_local.py`, `alpha_dist_plots.py`, `simp`, FGM, the trainer smoke blocks and the
  old `analysis/outputs/seed_sweep/*`. MIA is out of the pipeline:
  `analysis/privacy.py` (standalone, shrunk, tested) and `analysis/utils/runs.py`
  (`load_run_config`, `find_model_checkpoint`). Net −3 900 lines of py+yaml.
  Suite: 787 passed, 4 skipped.

## Next step: Phase 2 (one vocabulary, one metrics module, enforced schema)

See `plan.md` §2 Phase 2. Noticed along the way:
- `configs/trainer/adversarial/pgd_at.yaml` still has the stale `criterion` key and no
  `alpha`; schema enforcement (D25) will surface both.
- A training run fails unless its run dir has an `outputs/` ancestor
  (`experiments/tracking.py::_derive_group_key`).
- `analysis/utils/statistics.py`: `plot_accuracy_histogram`, `plot_mean_with_std`,
  `plot_scatter_vs_metric` are called nowhere (Phase 5).
- `analysis/utils/runs.py` still carries the legacy path-based config fallback
  (`_load_final_experiment_config`); D12 says new code doesn't read old run dirs.

## Settled just before hand-off (D42–D46, see `plan.md` Phase 4)

- **`spirals_capacity`** re-runs d10r6 with its own HPO as a consistency check against
  `spirals_nat` (D42).
- **AT efficiency:** PGD-AT at a reduced budget (5 training steps, 10 for validation, 40
  for evaluation, larger `eval_every`, smaller HPO budget). Fast AT / FGSM was
  rejected for rigour (D43), so the FGM branch is still deleted in Phase 1. A PGD-5 vs
  PGD-10 pilot on mnist12 d3r40 gates the step count before Phase 4.
- **`mnist_capacity`:** two pass criteria, α=0 near-best accuracy and an α=1 plateau
  (D44). `mnist_at` added (D46).
- **Default α ladder {0, 1e-3, 1e-2, 1e-1, 0.5, 1}**; 0.2 dropped (D45).

Still open: `spirals_embedding` α set ({0,1} or {1}); `mnist12_at` arches (r40 only or
also r20); the TS dataset list and grids (D6); the norm-control target expression for
cold MNIST runs (the pretrained model no longer supplies it: 0 or `n·ln d / 2`).

## Working with Martin

- He decides; Claude proposes. For each decision, give 2–3 options with a recommended
  one and push back when needed (he asks for it). He gives feedback in the option notes.
- **tensorkrowch first (D30):** no tensor-network maths in plain torch outside
  `core/model.py`. Workarounds become upstream candidates (`status.md` §12).
- Agent file: `AGENTS.md` committed, plus a one-line `CLAUDE.md` = `@AGENTS.md` (D27).
- Runs happen on the lab HPC: plain SSH, multi-GPU nodes, tmux, **no AI agents there**
  (D37), `BM4TC_DATA_ROOT=/ceph/chercheurs/nisseng261/bm4tc`. Tests run locally on the
  RTX 2080 eGPU (8 GB). The eGPU can be hot-plugged (headless NVIDIA 595 driver).

## Environment gotchas (this laptop)

- No bare `python`. Use `conda run --cwd $PWD -n bm4tc python -m pytest …` (plain
  `pytest` fails collection until `pytest.ini` exists). Baseline: 772 passed,
  4 skipped, 73 s.
- `coverage` is not in the env. It was installed into the session scratchpad with
  `pip --target`; reinstall the same way if needed, without touching the env.
- Hydra schema enforcement was verified by copying `configs/` to scratch and adding
  `- base_config` before `_self_` in `config.yaml`. It fails on the stale
  `trainer.adversarial.criterion` key first.
- `CLAUDE.md`, `DEFERRED.md` and `.claude/` are git-ignored on `main`. The other laptop
  may hold the old plan and issue list (C*/D* IDs) in them; reconcile on Friday.
