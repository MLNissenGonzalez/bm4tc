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
| `status.md` | survey of the codebase (§0–§10); **§11 = decisions D1–D41**; §12 = tensorkrowch upstream candidates | done; §11 is the source of truth for decisions |
| `plan.md` | target shape (§1), phases 0–9 (§2), untouched core (§3), sizes (§4), risks (§5), open questions (§6) | draft, Martin-approved apart from the open items below |
| `_paper/` (untracked, keep it untracked) | TPM 2026 paper (`main.tex`, outdated), NeurIPS rebuttal drafts | reference only |

`main` is untouched. Nothing has been pushed; ask before pushing.

## Next step: Phase 0 (safety net, no behaviour change)

See `plan.md` §2 Phase 0:
1. Tag `pre-ousterhout` on `main` (local; push only when Martin says).
2. Commit `pytest.ini` (`pythonpath = .`, `slow` marker). Un-ignore `pytest.ini`,
   `AGENTS.md` and `CLAUDE.md` in `.gitignore`.
3. Seam test `tests/e2e/test_pipeline.py`: CPU, < 60 s, tiny spirals, legendre d4r3,
   NAT α=0 → warm AT α=0.01 → analysis. It pins `objective`, clean acc, rob, purified
   acc and detection rate with tolerances. Only the harness function changes as entry
   points change.
4. Benchmark of training-step time (NAT and AT, d3r20, MNIST12-sized random data). This
   is the gate for D31 (cache ownership) in Phase 3.

## Open items Martin should settle (raised 2026-10-06, not yet answered)

On his Phase 4 study-grid edits (`plan.md` §2 Phase 4 table):
- **`spirals_capacity` "compare HPs with `spirals_nat`".** The meaning is unclear.
  - The (10,6) × α∈{0,1} cells duplicate `spirals_nat` cells; the proposal is to reuse
    those runs via manifests instead of re-training.
  - If "compare HPs" means *reuse* the d10r6 HPs at all capacities, that confounds
    capacity with tuning. The recommendation is HPO per arch.
- **`mnist12_nat` is now cold.** It is supported by `notebooks/coldvswarm.ipynb` (at r40
  α=1 cold wins, via its own lr). Consequence: the norm-control target can no longer come
  from the pretrained model, so it needs a fixed expression (0 or `n·ln d / 2`). AT still
  warm-starts from the α=0 NAT run.
- **`mnist12_at` "expensive, how cheaper?"** Options:
  - fewer PGD steps in training (10 → 3–5);
  - **fast AT** (FGSM with random start, Wong et al. 2020) during training, with PGD
    for validation and evaluation; this **conflicts with Phase 1 deleting the FGM
    branch**, so decide before Phase 1;
  - larger `eval_every`;
  - smaller HPO budget per cell.
- **`mnist_at` is missing** from the table. The main paper presumably needs AT on full
  MNIST.
- **`mnist_capacity`** needs concrete criteria (e.g. α=0 clean-acc plateau; α=1 test NLL
  / accuracy) and a grid (r values, maybe d). It must run before `mnist_nat`.
- `mnist_nat` uses α ∈ {0, 0.01, 0.1, 0.5, 1} (no 0.2, unlike `mnist12_nat`). Intended?
- `jem_*` mirrors MNIST and TS. JEM size must be matched to the MPS arch it is compared
  against (the rebuttal compared 349k-parameter JEM ≈ r20 against r40).

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
