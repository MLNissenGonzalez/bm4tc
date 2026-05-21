"""Visualize learned time-series distribution by sampling from a trained BornMachine.

Samples time-series curves per class and overlays them in one subplot per class,
with the y-axis in the original data scale (inverse-transformed from embedding range).

Usage
-----
    python -m analysis.visualize.ts_samples --run <run_dir>
    python -m analysis.visualize.ts_samples --run <run_dir> --num-spc 200 --num-bins 100
    python -m analysis.visualize.ts_samples --run <run_dir> --save-dir <dir>

NOTE: Sampling is not yet implemented for ConditionalBornMachine.
See DEFERRED.md § 'Canonical-form sampling' for the implementation path.
"""

raise NotImplementedError(
    "ts_samples.py requires BornGenerator.sample_all_classes(), which was removed in "
    "the CBM unification. See DEFERRED.md § 'Canonical-form sampling' for the path "
    "to re-implementing sampling with ConditionalBornMachine."
)
