# Pipeline Tools

Standalone maintenance scripts for the bm4tc experiment pipeline. Run from the project root; each script bootstraps `sys.path` automatically.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `delete_runs.py` | Delete sweep outputs: local dirs, W&B runs/artifacts, analysis dirs |

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
