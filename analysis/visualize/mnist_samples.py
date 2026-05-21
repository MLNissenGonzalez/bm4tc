"""Visualize learned MNIST distribution by sampling from a trained BornMachine.

Samples a few images per digit class and displays one randomly-chosen
sample per class in a 2×5 grid (digits 0–9).

Usage
-----
    python -m analysis.visualize.mnist_samples --run <run_dir>
    python -m analysis.visualize.mnist_samples --run <run_dir> --num-bins 100 --binarize
    python -m analysis.visualize.mnist_samples --run <run_dir> --save-dir <dir>

NOTE: Sampling is not yet implemented for ConditionalBornMachine.
See DEFERRED.md § 'Canonical-form sampling' for the implementation path.
"""

raise NotImplementedError(
    "mnist_samples.py requires BornGenerator.sample_all_classes(), which was removed in "
    "the CBM unification. See DEFERRED.md § 'Canonical-form sampling' for the path "
    "to re-implementing sampling with ConditionalBornMachine."
)
