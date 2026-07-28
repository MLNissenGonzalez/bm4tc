# Pipeline Tools

Standalone maintenance scripts for the bm4tc experiment pipeline. Run from the project root; each script bootstraps `sys.path` automatically.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `fill_hpo.py` | Patch `seed_sweep` configs with best HPO hyperparameters (W&B or local fallback) |
| `delete_runs.py` | Delete sweep outputs: local dirs, W&B runs/artifacts, analysis dirs |
| `migrate_configs.py` | One-off migration: old `nll/{dis,gen,mixed}/` layout → unified `{dataset}/{nat,at}/` layout |
| `alpha_lr_interp.py` | Compute geometrically-interpolated LRs for alpha-curve sweeps (historical, post-migration) |
| `fetcher.ipynb` | Interactive notebook for ad-hoc W&B data fetching |

---

## `fill_hpo.py` — Patch seed_sweep configs from HPO results

After a HPO run completes, propagate the best hyperparameters into the matching `seed_sweep` config.

```bash
# List all (hpo_kind → seed_kind) pairs and their fill status
python tools/fill_hpo.py --list

# Preview changes without writing (shows unified diff)
python tools/fill_hpo.py --dry-run
python tools/fill_hpo.py --trainer at --dry-run

# Apply to a specific combination
python tools/fill_hpo.py --dataset circles --embedding legendre

# Overwrite already-filled values
python tools/fill_hpo.py --force
```

**How it works**: scans `configs/experiments/` for `hpo*.yaml` files, finds the best run via W&B API (falling back to local Hydra outputs), and replaces `???  # FILL FROM HPO` placeholders in the corresponding `seed_sweep*.yaml`. Kind pairing is by stem: `hpo_a0 → seed_sweep_a0`, `hpo → seed_sweep`, etc.

---

## `delete_runs.py` — Delete sweep outputs

Removes a sweep's local outputs, W&B runs + artifacts, and mirrored `analysis/outputs/` directory.

```bash
# List all discovered sweep roots (no deletion)
python tools/delete_runs.py --list

# Preview what would be deleted (no confirmation prompt)
python tools/delete_runs.py --trainer nat --kind hpo_a0 --dry-run
python tools/delete_runs.py --dataset circles --date 2102 --dry-run
python tools/delete_runs.py --kind test --dry-run

# Delete (prompts for confirmation)
python tools/delete_runs.py --kind test
python tools/delete_runs.py --kind hpo --wandb-only --dry-run

# Clean up only analysis/outputs/ (when local + W&B are already gone)
python tools/delete_runs.py --embedding hermite --analysis-only --dry-run
python tools/delete_runs.py --analysis-only --list
```

Filter flags (`--trainer`, `--kind`, `--embedding`, `--arch`, `--dataset`, `--date`) all accept one or more values; they are OR-within a flag, AND-across flags.

---

## `migrate_configs.py` — Config layout migration (historical)

One-off script that renamed the old `nll/{dis,gen,mixed}/` + `adversarial/` layout to the unified `{dataset}/{nat|at}/` layout. Also strips per-file `hydra.sweep.dir` overrides that were superseded by the global template.

```bash
python tools/migrate_configs.py --dry-run    # print move plan
python tools/migrate_configs.py --execute    # git mv + strip overrides
```

The migration is already applied; keep this script for documentation and in case configs need to be re-migrated from an older branch.

---

## `alpha_lr_interp.py` — Geometric LR interpolation (historical)

Patches `alpha_curve_mixed` configs to use geometrically-interpolated learning rates across the alpha axis, expressed as OmegaConf resolver calls.

```bash
python tools/alpha_lr_interp.py            # print table + patch configs
python tools/alpha_lr_interp.py --dry-run  # print table only
```

Note: references the pre-migration `configs/experiments/generative/legendre/d10r6/alpha_curve_mixed/` path. Keep for reference; re-run only if rebuilding those configs from an old branch.
