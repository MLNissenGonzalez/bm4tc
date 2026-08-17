"""Transfer-attack + purification figure for MNIST CBMs.

One qualitative figure: rows are digit classes, columns are
``original | adversarial | purified <model 1> | purified <model 2> | purified <model 3>``.

1. Craft adversarial examples against the ATTACK SOURCE model (the adversarially-trained
   one) with a standard discriminative PGD attack at relative budget ``EPS_REL``.
2. Keep only examples on which the attack **transfers to every model**: all models classify
   the clean input correctly and all misclassify the adversarial one. Every purification
   column therefore starts from an identical, genuinely adversarial input.
3. Purify each transferring example with **each** model independently (likelihood
   purification within relative radius ``DELTA_REL``) and classify the purified input with the same
   model (self-purify, self-classify).
4. Emit a ``len(classes) x (2 + n_models)`` grid figure plus a plain-text statistics file
   holding the transferability and purification-success numbers over the whole eligible
   set (not just the handful of displayed digits).

The AT model has a poor generative model, so it should purify worse than the generatively
regularised models -- this script makes that contrast explicit on identical perturbations.

The undefended (alpha=0) model is included as a purification column: it is trained purely
discriminatively, so its density is a poor purifier and its column is the control.

CLI (each --models path may be a checkpoint file, a run dir, or a sweep root):
    python -m analysis.visualize.mnist_transfer_purify \
        --models 'a001=purified $\alpha=0.01$=outputs/.../seed_sweep/a001_2007/0/models/model' \
        --models 'at=AT-model purified=outputs/.../seed_sweep/a0_2507/0/models/model'

Notebook:
    from analysis.visualize.mnist_transfer_purify import transfer_purify_analysis
    fig, stats = transfer_purify_analysis()
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from analysis.utils import find_model_checkpoint, get_best_run, load_run_config
from src.analysis.purification import LikelihoodPurification
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.utils.embeddings import fmt_budget, range_size_of, rel_to_abs
from src.utils.evasion import ProjectedGradientDescent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG (defaults; override via CLI flags or function args) -------------------------
# (key, column label, path) in figure-column order. `path` is a sweep root (best run picked
# by clean accuracy), a numbered run dir, or a checkpoint file directly -- see
# `_resolve_run_dir`. All models must share the embedding (hence input_range); here all are
# legendre d3r20c64. a0 is the undefended baseline: discriminative, no purification-friendly
# density, so its purification column is the control.
MODELS: List[Tuple[str, str, str]] = [
    ("a0", r"purified $\alpha=0$ (undef.)",
     "outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep/a0_2406/0/models/model"),
    ("a001", r"purified $\alpha=0.01$",
     "outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep/a001_2007/0/models/model"),
    ("a01", r"purified $\alpha=0.1$",
     "outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep/a01_2107/0/models/model"),
    ("at", "AT-model purified",
     "outputs/mnist_full_r12/at/legendre/d3r20c64/seed_sweep/a0_2507/0/models/model"),
]
ATTACK_SOURCE_KEY = "at"   # attack is crafted white-box on this model
DISPLAY_CLASSES = [0, 3, 5, 9]  # one figure row per class

# Budgets are RELATIVE — fractions of the input domain width (see "Budget vocabulary"
# in CLAUDE.md). On legendre (width 2.0) both are absolute 0.2 in the model domain.
EPS_REL = 0.1             # attacker budget
DELTA_REL = 0.1           # purification radius (defense budget)
ATTACK_NUM_STEPS = 40
PURIFY_NUM_STEPS = 20
EVAL_BATCH_SIZE = 128
MAX_ATTACK_SAMPLES: Optional[int] = 2000  # #clean inputs attacked (None = whole test set)
STATS_CAP: Optional[int] = 200            # #eligible examples purified for the statistics
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "figures/mnist_r12/transfer_purify"

FIG_NAME = "transfer_purify_grid.pdf"
STATS_NAME = "transfer_purify_stats.txt"


def _out_names(eps_rel: float, delta_rel: float) -> Tuple[str, str]:
    """Output filenames stamped with the two budgets, so a sweep over ``--eps-rel`` /
    ``--delta-rel`` into one ``--save-dir`` does not overwrite itself.

    Budgets are **relative** (the authoring unit, see "Budget vocabulary" in CLAUDE.md);
    the ``rel`` in the stem says so, since the same number absolute means a different
    figure. ``fmt_budget`` keeps the stems free of float-repr noise.
    """
    suffix = f"epsrel{fmt_budget(eps_rel)}_deltarel{fmt_budget(delta_rel)}"
    return (f"{Path(FIG_NAME).stem}_{suffix}{Path(FIG_NAME).suffix}",
            f"{Path(STATS_NAME).stem}_{suffix}{Path(STATS_NAME).suffix}")


# --- Pure helpers (unit-tested; no model / no torch state) ------------------------------
def _transfer_mask(
    clean_preds: Sequence[np.ndarray],
    adv_preds: Sequence[np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """Boolean mask: keep examples **every** model classifies clean-correctly AND **every**
    model misclassifies adversarially (the attack transfers to all of them)."""
    labels = np.asarray(labels)
    mask = np.ones(len(labels), dtype=bool)
    for clean, adv in zip(clean_preds, adv_preds):
        mask &= np.asarray(clean) == labels
        mask &= np.asarray(adv) != labels
    return mask


def _select_rows(labels: np.ndarray, classes: Sequence[int]) -> Dict[int, Optional[int]]:
    """First index of each requested class in ``labels`` (None if the class is absent).

    Order of `classes` is preserved; `labels` is assumed to be in a deterministic
    (test-set) order, so the pick is reproducible and not cherry-picked.
    """
    labels = np.asarray(labels)
    out: Dict[int, Optional[int]] = {}
    for c in classes:
        hits = np.nonzero(labels == c)[0]
        out[int(c)] = int(hits[0]) if len(hits) else None
    return out


def _format_stats(stats: dict) -> str:
    """Render the statistics dict as the plain-text report."""
    keys: List[str] = list(stats["model_keys"])
    labels: Dict[str, str] = stats["model_labels"]
    src = stats["attack_source"]
    w = max(len(k) for k in keys) + 2

    L: List[str] = []
    L.append("MNIST transfer-attack + purification statistics")
    L.append("=" * 62)
    L.append("")
    L.append("--- config ---")
    for k in keys:
        mark = "  (attack source)" if k == src else ""
        L.append(f"  {k:<{w}} {stats['paths'][k]}{mark}")
        L.append(f"  {'':<{w}} label={labels[k]}  run={stats['run_dirs'][k]}")
    L.append(f"  attack        PGD-inf  eps_rel={stats['eps_rel']:.4g} "
             f"(abs {stats['abs_eps']:.4f}), {stats['attack_num_steps']} steps")
    L.append(f"  purification  likelihood-inf  delta_rel={stats['delta_rel']:.4g} "
             f"(abs {stats['abs_delta']:.4f}), {stats['purify_num_steps']} steps")
    L.append(f"  input range   [{stats['lo']:.4f}, {stats['hi']:.4f}]")
    L.append(f"  seed={stats['seed']}  device={stats['device']}  "
             f"batch_size={stats['eval_batch_size']}")
    L.append("")

    L.append("--- population ---")
    L.append(f"  test inputs attacked      {stats['n_attacked']}")
    L.append(f"  eligible (all models fooled, all clean-correct)  {stats['n_eligible']}")
    L.append("  eligible per class        " + ", ".join(
        f"{c}:{n}" for c, n in sorted(stats["eligible_per_class"].items())))
    L.append("")

    L.append("--- accuracy on the attacked subset ---")
    L.append(f"  {'model':<{w}} {'clean':>8} {'adv':>8} {'transfer':>10}")
    for k in keys:
        tr = stats["transfer_rate"][k]
        tr_s = "     n/a" if tr is None else f"{tr:>10.3f}"
        L.append(f"  {k:<{w}} {stats['clean_acc'][k]:>8.3f} {stats['adv_acc'][k]:>8.3f} {tr_s}")
    L.append(f"  transfer = fraction of examples that '{src}' and the model both classify "
             f"correctly when clean,")
    L.append(f"             and that the '{src}'-crafted perturbation flips for the model.")
    L.append("")

    L.append("--- purification success (self-purify, self-classify) ---")
    n_pur = stats["n_purified"]
    cap = stats["stats_cap"]
    note = ("" if cap is None or n_pur < cap
            else f"  [capped at --stats-cap {cap}, plus any forced figure rows]")
    L.append(f"  evaluated on {n_pur} of {stats['n_eligible']} eligible examples{note}")
    cls = sorted(stats["purify_per_class"][keys[0]].keys()) if keys else []
    L.append(f"  {'model':<{w}} {'overall':>8} " + " ".join(f"{'y=' + str(c):>7}" for c in cls))
    for k in keys:
        per = stats["purify_per_class"][k]
        cells = " ".join(
            ("      -" if per[c] is None else f"{per[c]:>7.3f}") for c in cls)
        L.append(f"  {k:<{w}} {stats['purify_acc'][k]:>8.3f} " + cells)
    L.append("  per-class counts          " + ", ".join(
        f"{c}:{n}" for c, n in sorted(stats["purified_per_class_n"].items())))
    L.append("")

    L.append("--- figure rows ---")
    for c, i in stats["row_index"].items():
        L.append(f"  y={c}: " + ("NO ELIGIBLE EXAMPLE" if i is None
                                 else f"eligible-set index {i}"))
    missing = [c for c, i in stats["row_index"].items() if i is None]
    if missing:
        L.append(f"  WARNING: classes {missing} have no transferring example; "
                 f"raise --max-attack-samples or --eps.")
    return "\n".join(L) + "\n"


# --- Model / data loading ---------------------------------------------------------------
def _resolve_run_dir(path: str) -> Tuple[Path, Optional[Path]]:
    """Resolve a --models path to (run_dir, checkpoint_or_None).

    Three accepted forms, so remote checkpoint paths can be pasted verbatim:
      * checkpoint file   ``.../<run>/models/model``  -> that exact checkpoint
      * run dir           ``.../<sweep>/0``           -> its models/ checkpoint
      * sweep root        ``.../<sweep>``             -> best run by clean accuracy
    A returned ``None`` checkpoint means "let find_model_checkpoint search run_dir".
    """
    p = Path(path)
    if p.is_file():
        # <run>/models/<ckpt>
        return p.parent.parent, p
    if (p / ".hydra").is_dir():
        return p, None
    # sweep root: analysis CSV mirrors outputs/ -> analysis/outputs/
    parts = list(p.parts)
    if "outputs" in parts:
        i = parts.index("outputs")
        csv = Path(*parts[:i], "analysis", *parts[i:]) / "evaluation_data.csv"
    else:
        csv = p / "evaluation_data.csv"
    if csv.exists():
        best = get_best_run(pd.read_csv(csv), "acc", minimize=False)
        return p / str(best["run_name"]), None
    logger.warning(f"No evaluation_data.csv at {csv}; using {p / '0'}")
    return p / "0", None


def _load_model(path: str, device: str) -> Tuple[ConditionalBornMachine, object, Path]:
    """Load a model from a checkpoint / run dir / sweep root; return (cbm, run_cfg, run_dir)."""
    run_dir, ckpt = _resolve_run_dir(path)
    run_cfg = load_run_config(run_dir)
    if ckpt is None:
        ckpt = find_model_checkpoint(run_dir)
    cbm = ConditionalBornMachine.load(str(ckpt), accumulate=True)
    cbm.to(device)
    cbm.eval()
    cbm.cache_log_Z()
    logger.info(f"Loaded {run_dir}")
    return cbm, run_cfg, run_dir


def _test_loader(run_cfg, cbm, batch_size: int):
    """Build the rescaled MNIST test loader from a run config (mirrors analysis/run.py)."""
    OmegaConf.update(run_cfg, "dataset.overwrite", True, force_add=True)
    dh = DataHandler(run_cfg.dataset)
    dh.load()
    dh.split_and_rescale(cbm)
    dh.get_classification_loaders()
    loader = dh.classification["test"]
    return torch.utils.data.DataLoader(loader.dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _classify(cbm: ConditionalBornMachine, x: torch.Tensor, device: str) -> np.ndarray:
    """Top-1 class predictions."""
    return cbm.class_probabilities(x.to(device)).argmax(dim=1).cpu().numpy()


# --- Main pipeline ----------------------------------------------------------------------
def transfer_purify_analysis(
    models: Sequence[Tuple[str, str, str]] = MODELS,
    attack_source: str = ATTACK_SOURCE_KEY,
    classes: Sequence[int] = DISPLAY_CLASSES,
    eps_rel: float = EPS_REL,
    delta_rel: float = DELTA_REL,
    attack_num_steps: int = ATTACK_NUM_STEPS,
    purify_num_steps: int = PURIFY_NUM_STEPS,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_attack_samples: Optional[int] = MAX_ATTACK_SAMPLES,
    stats_cap: Optional[int] = STATS_CAP,
    seed: int = SEED,
    device: str = DEVICE,
    save_dir: Optional[str] = SAVE_DIR,
) -> Tuple["object", dict]:
    """Run the transfer-attack + purification analysis.

    Returns (grid_figure, stats_dict). The stats dict is what `_format_stats` renders.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    keys = [k for k, _, _ in models]
    labels_map = {k: lbl for k, lbl, _ in models}
    paths = {k: pth for k, _, pth in models}
    if attack_source not in keys:
        raise ValueError(f"attack_source={attack_source!r} not among model keys {keys}")

    cbms: Dict[str, ConditionalBornMachine] = {}
    run_dirs: Dict[str, str] = {}
    src_cfg = None
    for k in keys:
        cbm, cfg, run_dir = _load_model(paths[k], device)
        cbms[k] = cbm
        run_dirs[k] = str(run_dir)
        if k == attack_source:
            src_cfg = cfg

    src_cbm = cbms[attack_source]
    # All models share the legendre embedding => identical input_range; build data once.
    loader = _test_loader(src_cfg, src_cbm, eval_batch_size)
    lo, hi = float(src_cbm.input_range[0]), float(src_cbm.input_range[1])
    # The single conversion boundary for this entry point.
    _range_size = range_size_of(src_cbm)
    abs_eps = rel_to_abs(eps_rel, _range_size)
    abs_delta = rel_to_abs(delta_rel, _range_size)
    logger.info(f"range=[{lo:.3f},{hi:.3f}] size={_range_size:.3f} | "
                f"eps_rel={eps_rel} -> abs {abs_eps:.4f} | "
                f"delta_rel={delta_rel} -> abs {abs_delta:.4f}")

    attack = ProjectedGradientDescent(norm="inf", num_steps=attack_num_steps, random_start=True)

    # --- Stage 1+2: attack the source, keep examples that fool every model ---------------
    kept_clean, kept_adv, kept_y = [], [], []
    kept_adv_pred: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    # running totals over ALL attacked inputs (not just eligible ones)
    n_attacked = 0
    n_clean_ok = {k: 0 for k in keys}
    n_adv_ok = {k: 0 for k in keys}
    n_pair_ok = {k: 0 for k in keys}      # both src and k clean-correct
    n_pair_flipped = {k: 0 for k in keys}  # ... and k flipped by the adv input

    for x, y in loader:
        if max_attack_samples is not None and n_attacked >= max_attack_samples:
            break
        n_attacked += len(x)
        x = x.to(device)
        y_np = y.numpy()

        clean_preds = {k: _classify(cbms[k], x, device) for k in keys}
        x_adv = attack.generate(src_cbm, x, y.to(device), abs_eps, device)
        adv_preds = {k: _classify(cbms[k], x_adv, device) for k in keys}

        src_clean_ok = clean_preds[attack_source] == y_np
        for k in keys:
            ck = clean_preds[k] == y_np
            n_clean_ok[k] += int(ck.sum())
            n_adv_ok[k] += int((adv_preds[k] == y_np).sum())
            pair = src_clean_ok & ck
            n_pair_ok[k] += int(pair.sum())
            n_pair_flipped[k] += int((pair & (adv_preds[k] != y_np)).sum())

        mask = _transfer_mask([clean_preds[k] for k in keys],
                              [adv_preds[k] for k in keys], y_np)
        if mask.any():
            idx = np.nonzero(mask)[0]
            kept_clean.append(x[idx].cpu())
            kept_adv.append(x_adv[idx].cpu())
            kept_y.append(y_np[idx])
            for k in keys:
                kept_adv_pred[k].append(adv_preds[k][idx])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not kept_y:
        raise RuntimeError("No transferring adversarial examples found. Try a stronger "
                           "attack (more steps / larger eps) or more attacked samples.")

    clean_imgs = torch.cat(kept_clean)
    adv_imgs = torch.cat(kept_adv)
    y_elig = np.concatenate(kept_y)
    adv_pred = {k: np.concatenate(kept_adv_pred[k]) for k in keys}
    n_eligible = len(y_elig)
    elig_per_class = {int(c): int((y_elig == c).sum()) for c in np.unique(y_elig)}
    logger.info(f"Attacked {n_attacked} inputs; {n_eligible} transfer to all models. "
                f"Per-class: {dict(sorted(elig_per_class.items()))}")

    row_index = _select_rows(y_elig, classes)
    missing = [c for c, i in row_index.items() if i is None]
    if missing:
        logger.warning(f"No transferring example for classes {missing}; "
                       f"those rows will be blank.")

    # --- Stage 3: purify with each model (self-purify, self-classify) --------------------
    # Cost cap: purify the first `stats_cap` eligible examples, but always include the
    # displayed rows so the figure never depends on the cap.
    if stats_cap is None or stats_cap >= n_eligible:
        sel = np.arange(n_eligible)
    else:
        forced = [i for i in row_index.values() if i is not None]
        sel = np.unique(np.concatenate([np.arange(stats_cap), np.asarray(forced, dtype=int)]))
        logger.info(f"Purifying {len(sel)} of {n_eligible} eligible examples (stats_cap).")
    pos_of = {int(i): p for p, i in enumerate(sel)}  # eligible index -> row in purified arrays
    sel_t = torch.as_tensor(sel, dtype=torch.long)
    adv_sel = adv_imgs[sel_t]

    purifier = LikelihoodPurification(norm="inf", num_steps=purify_num_steps, random_start=False)

    def _purify_all(cbm):
        """Purify the selected adv images (batched) with `cbm`; return (preds, images)."""
        preds, chunks = [], []
        for i in range(0, len(adv_sel), eval_batch_size):
            xb = adv_sel[i:i + eval_batch_size].to(device)
            xp, _ = purifier.purify(cbm, xb, abs_delta, device)
            preds.append(_classify(cbm, xp, device))
            chunks.append(xp.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(preds), torch.cat(chunks)

    pur_pred: Dict[str, np.ndarray] = {}
    pur_imgs: Dict[str, torch.Tensor] = {}
    for k in keys:
        pur_pred[k], pur_imgs[k] = _purify_all(cbms[k])

    y_sel = y_elig[sel]
    success = {k: pur_pred[k] == y_sel for k in keys}
    sel_classes = sorted(int(c) for c in np.unique(y_sel))
    for k in keys:
        logger.info(f"Purification success [{k}]: {success[k].mean():.3f}")

    stats = {
        "model_keys": keys,
        "model_labels": labels_map,
        "paths": paths,
        "run_dirs": run_dirs,
        "attack_source": attack_source,
        "eps_rel": eps_rel, "abs_eps": abs_eps, "attack_num_steps": attack_num_steps,
        "delta_rel": delta_rel, "abs_delta": abs_delta, "purify_num_steps": purify_num_steps,
        "lo": lo, "hi": hi, "seed": seed, "device": str(device),
        "eval_batch_size": eval_batch_size,
        "n_attacked": n_attacked,
        "n_eligible": n_eligible,
        "eligible_per_class": elig_per_class,
        "clean_acc": {k: n_clean_ok[k] / n_attacked for k in keys},
        "adv_acc": {k: n_adv_ok[k] / n_attacked for k in keys},
        "transfer_rate": {
            k: (n_pair_flipped[k] / n_pair_ok[k]) if n_pair_ok[k] else None for k in keys},
        "stats_cap": stats_cap,
        "n_purified": int(len(sel)),
        "purify_acc": {k: float(success[k].mean()) for k in keys},
        "purify_per_class": {
            k: {c: (float(success[k][y_sel == c].mean()) if (y_sel == c).any() else None)
                for c in sel_classes}
            for k in keys},
        "purified_per_class_n": {c: int((y_sel == c).sum()) for c in sel_classes},
        "row_index": row_index,
    }

    fig = _plot_grid(
        models, classes, row_index, pos_of, clean_imgs, adv_imgs, pur_imgs,
        y_elig, adv_pred[attack_source], pur_pred, lo, hi,
    )

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_name, stats_name = _out_names(eps_rel, delta_rel)
        fig.savefig(out / fig_name, dpi=300, bbox_inches="tight")
        (out / stats_name).write_text(_format_stats(stats))
        logger.info(f"Saved {out / fig_name} and {out / stats_name}")

    return fig, stats


# --- Plotting ---------------------------------------------------------------------------
def _plot_grid(
    models, classes, row_index, pos_of, clean, adv, pur_imgs,
    true_labels, attack_predictions, purify_predictions, lo, hi,
):
    """Grid: one row per class, columns [original | adversarial | one per model].

    Each panel includes its model prediction as a small caption below the image.
    """
    import matplotlib.pyplot as plt

    keys = [k for k, _, _ in models]
    col_titles = ["original", "adversarial"] + [lbl for _, lbl, _ in models]
    img_dim = math.isqrt(clean.shape[1])

    def _img(t):
        return ((t.float() - lo) / (hi - lo)).reshape(img_dim, img_dim).numpy()

    n_rows, n_cols = len(classes), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols, 1.75 * n_rows))
    axes = np.atleast_2d(axes)

    for r, c in enumerate(classes):
        i = row_index[int(c)]
        for j in range(n_cols):
            ax = axes[r, j]
            ax.set_xticks([]); ax.set_yticks([])
            if i is None:
                ax.set_facecolor("0.9")
                if j == 0:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="0.4")
                continue
            if j == 0:
                img = _img(clean[i])
                panel_prediction = int(true_labels[i])
            elif j == 1:
                img = _img(adv[i])
                panel_prediction = int(attack_predictions[i])
            else:
                k, p = keys[j - 2], pos_of[int(i)]
                img = _img(pur_imgs[k][p])
                panel_prediction = int(purify_predictions[k][p])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel(
                f"Pred. {panel_prediction}", fontsize=8, color="0.35", labelpad=2
            )
        prediction = "n/a" if i is None else str(int(attack_predictions[i]))
        axes[r, 0].set_ylabel(
            f"Class {int(c)}\n(Pred. {prediction})",
            fontsize=10, rotation=0, ha="right", va="center", labelpad=42,
        )

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    fig.tight_layout()
    return fig


def _parse_models(specs: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    """Turn ``--models key=path`` (or ``key=label=path``) specs into the MODELS triples.

    The key is everything before the first ``=`` and the path everything after the last, so
    labels may themselves contain ``=`` (mathtext such as ``$\\alpha=0.01$``).
    """
    if not specs:
        return MODELS
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--models expects key=path or key=label=path, got {s!r}")
        key, rest = s.split("=", 1)
        if "=" in rest:
            label, path = rest.rsplit("=", 1)
        else:
            label, path = key, rest
        if not key or not path:
            raise ValueError(f"--models expects key=path or key=label=path, got {s!r}")
        out.append((key, label, path))
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", action="append", default=None,
                   help="key=path or key=label=path; repeat once per column (in order). "
                        "path = checkpoint file, run dir, or sweep root")
    p.add_argument("--attack-source", default=ATTACK_SOURCE_KEY)
    p.add_argument("--classes", default=",".join(str(c) for c in DISPLAY_CLASSES))
    p.add_argument("--eps-rel", type=float, default=EPS_REL)
    p.add_argument("--delta-rel", type=float, default=DELTA_REL)
    p.add_argument("--attack-num-steps", type=int, default=ATTACK_NUM_STEPS)
    p.add_argument("--purify-num-steps", type=int, default=PURIFY_NUM_STEPS)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--max-attack-samples", type=int, default=MAX_ATTACK_SAMPLES)
    p.add_argument("--stats-cap", type=int, default=STATS_CAP)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--save-dir", default=SAVE_DIR)
    a = p.parse_args()

    transfer_purify_analysis(
        models=_parse_models(a.models), attack_source=a.attack_source,
        classes=[int(c) for c in a.classes.split(",")],
        eps_rel=a.eps_rel, delta_rel=a.delta_rel, attack_num_steps=a.attack_num_steps,
        purify_num_steps=a.purify_num_steps, eval_batch_size=a.eval_batch_size,
        max_attack_samples=a.max_attack_samples, stats_cap=a.stats_cap,
        seed=a.seed, device=a.device, save_dir=a.save_dir,
    )
