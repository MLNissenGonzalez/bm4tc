"""Transfer attack and score-purification grid for the MNIST JEM models.

The grid uses exactly the same eligible examples in every purification column:
an AT-source PGD perturbation that transfers to every selected model.  Each
model then score-purifies that common adversarial input independently.

Run from the project root:

    python -m baselines.jem.transfer_purify --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.datahandler import DataHandler

from .analysis import load_run_config
from .attacks import PGDConfig, pgd_classification
from .device import resolve_device
from .model import JEMMLP
from .purification import PurificationConfig, gradient_purify

logger = logging.getLogger(__name__)

ModelSpec = tuple[str, str, str]

MODELS: list[ModelSpec] = [
    (
        "a0",
        r"purified $\alpha=0$ (undef.)",
        "outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a0_2607/0",
    ),
    (
        "a001",
        r"purified $\alpha=0.01$",
        "outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a001_2607/0",
    ),
    (
        "a01",
        r"purified $\alpha=0.1$",
        "outputs/baselines/jem/mnist_full_r12/nat/mlp_h550x480/seed_sweep/a01_2607/0",
    ),
    (
        "at",
        "AT-model purified",
        "outputs/baselines/jem/mnist_full_r12/at/mlp_h550x480/seed_sweep/at_2707/0",
    ),
]
ATTACK_SOURCE_KEY = "at"
DISPLAY_CLASSES = [0, 3, 5, 9]

EPS = 0.2
RADIUS = 0.3
ATTACK_NUM_STEPS = 40
PURIFY_NUM_STEPS = 20
EVAL_BATCH_SIZE = 128
MAX_ATTACK_SAMPLES: Optional[int] = 2000
STATS_CAP: Optional[int] = 200
SEED = 0
SAVE_DIR = "figures/jem_mnist/transfer_purify"
FIG_NAME = "transfer_purify_grid.pdf"
STATS_NAME = "transfer_purify_stats.txt"


def _transfer_mask(
    clean_preds: Sequence[np.ndarray], adv_preds: Sequence[np.ndarray], labels: np.ndarray
) -> np.ndarray:
    """Keep inputs every model gets right clean and wrong after the transfer attack."""
    mask = np.ones(len(labels), dtype=bool)
    for clean, adv in zip(clean_preds, adv_preds):
        mask &= np.asarray(clean) == labels
        mask &= np.asarray(adv) != labels
    return mask


def _select_rows(labels: np.ndarray, classes: Sequence[int]) -> dict[int, Optional[int]]:
    """Select the first eligible image for each requested digit class."""
    return {
        int(class_idx): (
            int(hits[0]) if len(hits := np.flatnonzero(labels == class_idx)) else None
        )
        for class_idx in classes
    }


def _run_name(value) -> str:
    """Render CSV run names without turning an integer seed into ``0.0``."""
    return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


def _resolve_run_dir(path: str) -> Path:
    """Accept a checkpoint, a numbered JEM run, or a seed-sweep root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.parent.parent
    if (candidate / "models/model.pt").exists():
        return candidate

    parts = list(candidate.parts)
    if "outputs" in parts:
        marker = parts.index("outputs")
        csv = Path(*parts[:marker], "analysis", *parts[marker:]) / "evaluation_data.csv"
    else:
        csv = candidate / "evaluation_data.csv"
    if csv.exists():
        results = pd.read_csv(csv)
        best = results.loc[results["acc"].idxmax()]
        return candidate / _run_name(best["run_name"])
    logger.warning("No evaluation CSV at %s; using run 0 below %s", csv, candidate)
    return candidate / "0"


def _load_model(path: str, device: torch.device):
    run_dir = _resolve_run_dir(path)
    model, _ = JEMMLP.load(run_dir / "models/model.pt", device=device)
    model.eval()
    return model, load_run_config(run_dir), run_dir


def _test_loader(cfg, model: JEMMLP, batch_size: int):
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(model)
    datahandler.get_classification_loaders(batch_size=batch_size)
    return datahandler.classification["test"]


@torch.no_grad()
def _classify(model: JEMMLP, data: torch.Tensor) -> np.ndarray:
    return model(data).argmax(dim=1).cpu().numpy()


def _format_stats(stats: dict) -> str:
    keys = stats["model_keys"]
    width = max(len(key) for key in keys) + 2
    lines = ["JEM MNIST transfer-attack + purification statistics", "=" * 62, ""]
    lines += ["--- config ---"]
    for key in keys:
        marker = "  (attack source)" if key == stats["attack_source"] else ""
        lines.append(f"  {key:<{width}} {stats['paths'][key]}{marker}")
        lines.append(
            f"  {'':<{width}} label={stats['model_labels'][key]}  run={stats['run_dirs'][key]}"
        )
    lines += [
        f"  attack        PGD-inf eps={stats['eps']} (rel) = {stats['abs_eps']:.4f} (abs), "
        f"{stats['attack_num_steps']} steps",
        f"  purification  score-ascent-inf radius={stats['radius']} (rel) = "
        f"{stats['abs_radius']:.4f} (abs), {stats['purify_num_steps']} steps",
        f"  input range   [{stats['lo']:.4f}, {stats['hi']:.4f}]",
        "",
        "--- population ---",
        f"  test inputs attacked      {stats['n_attacked']}",
        f"  eligible (all models fooled, all clean-correct)  {stats['n_eligible']}",
        "  eligible per class        " + ", ".join(
            f"{label}:{count}" for label, count in sorted(stats["eligible_per_class"].items())
        ),
        "",
        "--- purification success (self-purify, self-classify) ---",
        f"  evaluated on {stats['n_purified']} of {stats['n_eligible']} eligible examples",
        f"  {'model':<{width}} {'overall':>8}",
    ]
    for key in keys:
        lines.append(f"  {key:<{width}} {stats['purify_acc'][key]:>8.3f}")
    lines += ["", "--- figure rows ---"]
    for label, index in stats["row_index"].items():
        lines.append(
            f"  y={label}: " + ("NO ELIGIBLE EXAMPLE" if index is None else f"eligible-set index {index}")
        )
    return "\n".join(lines) + "\n"


def _plot_grid(
    models: Sequence[ModelSpec],
    classes: Sequence[int],
    row_index: dict[int, Optional[int]],
    selected_position: dict[int, int],
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    purified: dict[str, torch.Tensor],
    input_range: tuple[float, float],
):
    keys = [key for key, _, _ in models]
    titles = ["original", "adversarial"] + [label for _, label, _ in models]
    image_dim = math.isqrt(clean.shape[1])
    if image_dim * image_dim != clean.shape[1]:
        raise ValueError("Transfer-purity grid requires square, flattened images.")
    lo, hi = input_range

    def to_image(data: torch.Tensor):
        return ((data.float() - lo) / (hi - lo)).reshape(image_dim, image_dim).numpy()

    figure, axes = plt.subplots(
        len(classes), len(titles), figsize=(1.55 * len(titles), 1.75 * len(classes)), squeeze=False
    )
    for row, class_idx in enumerate(classes):
        eligible_index = row_index[int(class_idx)]
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            if eligible_index is None:
                axis.set_facecolor("0.9")
                if column == 0:
                    axis.text(0.5, 0.5, "n/a", ha="center", va="center", transform=axis.transAxes)
                continue
            if column == 0:
                image = to_image(clean[eligible_index])
            elif column == 1:
                image = to_image(adversarial[eligible_index])
            else:
                image = to_image(purified[keys[column - 2]][selected_position[int(eligible_index)]])
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"$y={int(class_idx)}$", fontsize=11)
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def transfer_purify_analysis(
    models: Sequence[ModelSpec] = MODELS,
    attack_source: str = ATTACK_SOURCE_KEY,
    classes: Sequence[int] = DISPLAY_CLASSES,
    eps: float = EPS,
    radius: float = RADIUS,
    attack_num_steps: int = ATTACK_NUM_STEPS,
    purify_num_steps: int = PURIFY_NUM_STEPS,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_attack_samples: Optional[int] = MAX_ATTACK_SAMPLES,
    stats_cap: Optional[int] = STATS_CAP,
    seed: int = SEED,
    device: str = "auto",
    save_dir: Optional[str | Path] = SAVE_DIR,
):
    """Create the JEM transfer-purification grid and its statistics report."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = resolve_device(device)
    keys = [key for key, _, _ in models]
    if attack_source not in keys:
        raise ValueError(f"attack_source={attack_source!r} is not one of {keys}")

    loaded, paths, run_dirs, source_cfg = {}, {}, {}, None
    for key, _, path in models:
        model, cfg, run_dir = _load_model(path, resolved_device)
        loaded[key] = model
        paths[key] = path
        run_dirs[key] = str(run_dir)
        if key == attack_source:
            source_cfg = cfg
    source = loaded[attack_source]
    loader = _test_loader(source_cfg, source, eval_batch_size)
    lo, hi = map(float, source.input_range)
    absolute_eps, absolute_radius = eps * (hi - lo), radius * (hi - lo)

    clean_parts, adversarial_parts, label_parts = [], [], []
    attacked = 0
    for clean, labels in loader:
        if max_attack_samples is not None and attacked >= max_attack_samples:
            break
        clean, labels = clean.to(resolved_device), labels.to(resolved_device)
        if max_attack_samples is not None:
            remaining = max_attack_samples - attacked
            clean, labels = clean[:remaining], labels[:remaining]
        if not len(clean):
            break
        attacked += len(clean)
        clean_predictions = {key: _classify(model, clean) for key, model in loaded.items()}
        adversarial = pgd_classification(
            source, clean, labels, PGDConfig(epsilon=absolute_eps, num_steps=attack_num_steps)
        )
        adversarial_predictions = {
            key: _classify(model, adversarial) for key, model in loaded.items()
        }
        mask = _transfer_mask(
            [clean_predictions[key] for key in keys],
            [adversarial_predictions[key] for key in keys],
            labels.cpu().numpy(),
        )
        if mask.any():
            selected = np.flatnonzero(mask)
            clean_parts.append(clean[selected].cpu())
            adversarial_parts.append(adversarial[selected].cpu())
            label_parts.append(labels[selected].cpu().numpy())

    if not label_parts:
        raise RuntimeError("No adversarial inputs transferred to every model.")
    clean = torch.cat(clean_parts)
    adversarial = torch.cat(adversarial_parts)
    labels = np.concatenate(label_parts)
    row_index = _select_rows(labels, classes)
    eligible = len(labels)
    if stats_cap is None or stats_cap >= eligible:
        selected = np.arange(eligible)
    else:
        required = [index for index in row_index.values() if index is not None]
        selected = np.unique(np.concatenate((np.arange(stats_cap), required)))
    selected_position = {int(index): position for position, index in enumerate(selected)}
    selected_adversarial = adversarial[torch.as_tensor(selected)]
    purification_cfg = PurificationConfig(radius=absolute_radius, num_steps=purify_num_steps)

    purified, purified_predictions = {}, {}
    for key, model in loaded.items():
        chunks = []
        for start in range(0, len(selected_adversarial), eval_batch_size):
            batch = selected_adversarial[start : start + eval_batch_size].to(resolved_device)
            chunks.append(gradient_purify(model, batch, purification_cfg).cpu())
        purified[key] = torch.cat(chunks)
        purified_predictions[key] = _classify(model, purified[key].to(resolved_device))
    selected_labels = labels[selected]
    success = {key: prediction == selected_labels for key, prediction in purified_predictions.items()}
    stats = {
        "model_keys": keys,
        "model_labels": {key: label for key, label, _ in models},
        "paths": paths,
        "run_dirs": run_dirs,
        "attack_source": attack_source,
        "eps": eps,
        "abs_eps": absolute_eps,
        "attack_num_steps": attack_num_steps,
        "radius": radius,
        "abs_radius": absolute_radius,
        "purify_num_steps": purify_num_steps,
        "lo": lo,
        "hi": hi,
        "n_attacked": attacked,
        "n_eligible": eligible,
        "eligible_per_class": {int(c): int((labels == c).sum()) for c in np.unique(labels)},
        "n_purified": len(selected),
        "purify_acc": {key: float(value.mean()) for key, value in success.items()},
        "row_index": row_index,
    }
    figure = _plot_grid(
        models, classes, row_index, selected_position, clean, adversarial, purified, (lo, hi)
    )
    if save_dir is not None:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output / FIG_NAME, dpi=300, bbox_inches="tight")
        (output / STATS_NAME).write_text(_format_stats(stats))
        logger.info("Saved %s and %s", output / FIG_NAME, output / STATS_NAME)
    return figure, stats


def _parse_models(specs: Optional[list[str]]) -> Sequence[ModelSpec]:
    if not specs:
        return MODELS
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--models expects key=path or key=label=path, got {spec!r}")
        key, rest = spec.split("=", 1)
        label, path = rest.rsplit("=", 1) if "=" in rest else (key, rest)
        if not key or not path:
            raise ValueError(f"--models expects key=path or key=label=path, got {spec!r}")
        parsed.append((key, label, path))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", action="append")
    parser.add_argument("--attack-source", default=ATTACK_SOURCE_KEY)
    parser.add_argument("--classes", default=",".join(map(str, DISPLAY_CLASSES)))
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument("--attack-num-steps", type=int, default=ATTACK_NUM_STEPS)
    parser.add_argument("--purify-num-steps", type=int, default=PURIFY_NUM_STEPS)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--max-attack-samples", type=int, default=MAX_ATTACK_SAMPLES)
    parser.add_argument("--stats-cap", type=int, default=STATS_CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    args = parser.parse_args()
    transfer_purify_analysis(
        models=_parse_models(args.models),
        attack_source=args.attack_source,
        classes=[int(value) for value in args.classes.split(",")],
        eps=args.eps,
        radius=args.radius,
        attack_num_steps=args.attack_num_steps,
        purify_num_steps=args.purify_num_steps,
        eval_batch_size=args.eval_batch_size,
        max_attack_samples=args.max_attack_samples,
        stats_cap=args.stats_cap,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
