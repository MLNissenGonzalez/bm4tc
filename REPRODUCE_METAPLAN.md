# Reproduction Notebook — Meta-Plan

**Goal**: A top-level `reproduce_figures.ipynb` that lets a reader reproduce the three main paper figures (spirals) in the shortest path through the repo.

---

## Target Figures

| # | Paper label | File in paper | What it shows |
|---|-------------|---------------|---------------|
| 1 | `fig:alpha-density` | `dists_with_adv.png` | Panel of p(c\|x) + p(x) heatmaps for at, α=0, α=0.5, α=1 |
| 2 | `fig:alpha-robustness` | `legendre_d10D6_2804_eps0.2.pdf` | Accuracy/robustness + NLL vs α (alpha-curve line plot) |
| 3 | `fig:regime-spirals` | `metric_eps02.pdf` | Barplot: clean/rob/accepted/purified for 4 regimes at ε=0.2 |

All figures use: **spirals** dataset, **Legendre** embedding, **d10r6** arch.

---

## Required Runs (spirals / legendre / d10r6)

| Kind | Config | Seeds | Used by figs |
|------|--------|-------|-------------|
| `seed_sweep_a0` | `spirals/nat/legendre/d10r6/seed_sweep_a0.yaml` | 20 | 1, 2 (endpoint), 3 |
| `seed_sweep_a05` | `spirals/nat/legendre/d10r6/seed_sweep_a05.yaml` | 20 | 3 |
| `seed_sweep_a1` | `spirals/nat/legendre/d10r6/seed_sweep_a1.yaml` | 20 | 1, 2 (endpoint), 3 |
| `seed_sweep` (at) | `spirals/at/legendre/d10r6/seed_sweep.yaml` | 8 | 1, 3 |
| `alpha_curve` | `spirals/nat/legendre/d10r6/alpha_curve.yaml` | 5 × N_α | 2 |

Configs are pre-filled with HPOs. If re-running HPO from scratch, see Section 0 of the notebook.

---

## Notebook Structure

```
reproduce_figures.ipynb
  §0  [Optional] Re-run HPO + fill HPs
  §1  Run seed sweeps  (terminal commands + subprocess cells)
  §2  Run post-hoc analysis  (analysis/sweep.py for each sweep)
  §3  Figure 1 — Distribution panel  (assemble PNGs from analysis outputs)
  §4  Figure 2 — Alpha-curve line plot  (adapted from reference/alpha_curve_plot.py)
  §5  Figure 3 — Regime barplot  (adapted from reference/toy_analysis.py §14)
```

All figure code is inline; no external `reference/` dependency.

---

## Phase Breakdown (sessions)

### Phase 1 — This session: Meta-plan + memory  ✓
**Outputs**: `REPRODUCE_METAPLAN.md`, memory entry  
**Blockers**: none

---

### Phase 2 — Investigation + notebook skeleton
**Goal**: Create bare notebook skeleton; resolve remaining open question about distribution PNG content.

**Pre-answered (resolved in Phase 1 session)**:
- ✓ **HPO fill status**: All 4 seed sweeps have numeric `lr`/`weight_decay` filled. `at/seed_sweep` also has `clean_weight` filled. No manual fills needed before running.
- ✓ **`alpha_curve` structure**: IS a multirun — 10 α values × 5 seeds = 50 numbered subdirs.  
  α values: `0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 0.8, 1.0`. LR interpolated via `geom_lr` resolver (no fill needed).
- ✓ **Analysis output path**: `outputs/spirals/nat/legendre/d10r6/{kind}_{DDMM}` → `analysis/outputs/spirals/nat/legendre/d10r6/{kind}_{DDMM}/`. Confirmed via sweep.py line 330.
- ✓ **CSV robustness column**: `_STRENGTH_FRACTIONS = [0.05, 0.1, 0.15]` × Legendre range (2.0) → `eval/test/rob/0.2` for ε=0.2. ✓
- ✓ **COMPUTE_DISTRIBUTIONS default**: `False` in sweep.py. No `--viz` CLI flag exists — must edit the config block to `True` before running if distribution PNGs are needed (Fig 1). Note this prominently in §2.

**All open questions resolved** (see findings below). Phase 2 is now just creating the skeleton.

**Task**: Create `reproduce_figures.ipynb` skeleton: section headers (markdown cells), placeholder `# TODO` code cells.

**Output**: `reproduce_figures.ipynb` with skeleton.

---

### Phase 3 — Training + analysis sections (§0–§2)
**Goal**: Fill in the training and analysis cells/markdown.

**Tasks**:
1. §0 (optional HPO): Note on `hpo_a0/a05/a1` configs; how to run `fill_hpo.py` or set by hand.
2. §1 (seed sweeps): Terminal + subprocess commands for all 5 sweeps; tmux tip for parallelism.
3. §2 (analysis): `analysis/sweep.py` commands for each sweep dir; note on `COMPUTE_DISTRIBUTIONS`.

**Depends on**: Phase 2 (exact paths confirmed).

**Output**: §0–§2 complete in notebook.

---

### Phase 4 — Figure code (§3–§5)  ✓
**Goal**: Port and adapt all three figure generation routines as clean notebook cells.

**§3 — Fig 1 (distribution panel)**:
- Strategy: run sweep.py with `COMPUTE_DISTRIBUTIONS=True` for the 4 sweeps → it saves `decision_boundary.png` + `best_joint.png` per sweep via `visualize_from_run_dir`.
- Notebook cell: load saved PNGs from the 4 analysis output dirs, assemble 2×4 grid with matplotlib (no model reloading).
- Colormap: `distributions.py` uses blue→white→orange diverging map for p(c|x) ✓ (matches paper).
- Column ordering: [at, a0, a05, a1].
- Output: `figures/dists_with_adv.png`
- Note: `alpha_dist_plots.py` is NOT used here — it's for the alpha_curve sweep only; Fig 1 needs best runs from 4 separate seed sweeps.

**§4 — Fig 2 (alpha curve)**:
- Source: `analysis/visualize/alpha_curve_plots.py` — already in codebase, reads alpha_curve analysis CSV.
- Hardcoded for Legendre ε=0.2 (correct for spirals). Already uses symlog x-axis matching paper.
- **Missing vs. paper**: no endpoint override (paper used 20-seed dedicated sweeps at α=0,1; `alpha_curve_plots.py` uses only the 5-seed alpha_curve runs at those endpoints). Decision for Phase 4: use as-is (simpler, minor accuracy difference) or add override logic.
- Usage: `python analysis/visualize/alpha_curve_plots.py <alpha_curve_analysis_dir>` — saves `alpha_curve.png` alongside CSV.
- Notebook cell: just call this script (or inline a trimmed version). Output: `figures/spirals/alpha/alpha_curve.png`

**§5 — Fig 3 (regime barplot)**:
- Source: `reference/toy_analysis.py` `cell14_metric_barplots()` — adapt for `nat/at` paths.
- Old paths: `analysis/outputs/seed_sweep/{cls|gen|adv}/legendre/d10D6/spirals_4k_{date}/`
- New paths: `analysis/outputs/spirals/{nat|at}/legendre/d10r6/{kind}_{DDMM}/`
- Models: at → `at/seed_sweep`, α=0 → `nat/seed_sweep_a0`, α=0.5 → `nat/seed_sweep_a05`, α=1 → `nat/seed_sweep_a1`
- CSV columns already confirmed: `eval/test/rob/0.2`, `eval/uq_purify_acc/0.2/0.2`, `eval/gibbs_purify_acc/0.2/1`, `eval/uq_det_err_passed/10pct/0.2`
- Output: `figures/spirals/regime/metric_eps02.pdf`

**Depends on**: Phase 2 (confirmed paths + CSV schema), Phase 3 (analysis outputs exist or are described).

**Output**: §3–§5 complete in notebook.

---

### Phase 5 — Verify + polish
**Goal**: End-to-end check; clean up prose and cell structure.

**Tasks**:
1. Run §3–§5 against existing `analysis/outputs/` data (old paths if new runs not yet available).
2. Fix path issues; ensure figures render correctly.
3. Add brief inline comments in figure code cells.
4. Final pass: simplicity check — no cell should be > ~60 lines; split if needed.

**Output**: `reproduce_figures.ipynb` ready to ship.

---

## Findings (resolved before Phase 2)

### `analysis/visualize/distributions.py`
- `plot_decision_boundary`: diverging blue→white→orange colormap, white band at 0.5 boundary — matches paper Fig 1 ✓
- `plot_joint_marginal`: "Purples" colormap, class-normalized marginal
- `visualize_from_run_dir`: high-level entry point; saves `decision_boundary.png` + `best_joint.png` (not `best_class_dist.png` — the GUIDE.md had a stale filename)
- Called by sweep.py when `COMPUTE_DISTRIBUTIONS=True`

### `analysis/visualize/alpha_curve_plots.py`
- Current-codebase Fig 2 script. Reads `evaluation_data.csv` from alpha_curve analysis dir.
- Hardcoded for Legendre ε=0.2 (appropriate for spirals). Uses symlog x-axis.
- **STALE (separate session TODO)**: column constants use old `eval/test/` prefix format
  (`eval/test/acc`, `eval/test/rob/0.2`, etc.), but the current `analysis/sweep.py` outputs
  columns WITHOUT that prefix (`acc`, `rob/0.2`, `dis_loss`, `gen_loss`, ...). The script
  silently produces empty plots when run against new CSVs. Notebook cell 11 inlines a corrected
  version; `alpha_curve_plots.py` itself needs updating as a standalone script.
- **Gap resolved in Phase 4**: notebook cell 11 overrides α=0,1 endpoints with 20-seed
  seed_sweep_a0/a1 data.

### `analysis/visualize/alpha_dist_plots.py`
- Loads models from an alpha_curve sweep and renders p(c|x)/p(x) for each α.
- **Not used for Fig 1** — Fig 1 needs best runs from 4 separate seed sweeps (at, a0, a05, a1). Use `visualize_from_run_dir` directly for those.
- Useful for exploratory/appendix purposes (distribution evolution across α) but not one of the 3 main figures.

---

## Cross-Phase Notes

- **Path convention**: Old analysis outputs use `cls/gen/adv` + `d10D6` (capital D). New outputs will use `nat/at` + `d10r6`. Figure code in Phase 4 must target new convention, but may need a fallback to old paths for testing during Phase 5.
- **`alpha_curve` config name**: Currently `configs/experiments/spirals/nat/legendre/d10r6/alpha_curve.yaml` — may be renamed before Phase 3; check before hardcoding.
- **Seed counts**: α=0, α=0.5, α=1 → 20 seeds; at → 8 seeds; alpha_curve intermediate → 5 seeds.
- **eps convention**: Legendre ε=0.2 in the paper = 0.1 × range_size (range=2). Analysis uses absolute values; confirm CSV column is `eval/test/rob/0.2`.
- **LaTeX in matplotlib**: Figure code uses `text.usetex=True`. If the reader has no LaTeX install, add a graceful fallback toggle.
