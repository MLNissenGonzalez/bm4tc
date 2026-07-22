"""Transfer-attack + differential-purification visualisation for MNIST CBMs.

Compares an adversarially-trained (AT) model against a generatively-regularised model
on the *same* adversarial inputs:

1. Craft adversarial examples against the SOURCE model (default AT) with a standard
   discriminative PGD attack at strength ``EPS`` (range-relative fraction).
2. Keep only examples that **fool both** models (both classify the clean input correctly
   and both misclassify the adversarial one) -- i.e. the attack transfers. No joint loss.
3. Purify each transferring example with **each** model independently (likelihood
   purification within radius ``RADIUS``), classify the purified input with the same
   model, and record whether the correct label is recovered (self-purify, self-classify).
4. Track, per example, which model purifies successfully (categories: both / source-only /
   target-only / neither) and dump a per-example CSV + a summary figure + an example montage.

The AT model has a poor generative model, so it should purify worse than the generative
model -- this script makes that contrast explicit on identical perturbations.

CLI:
    python -m analysis.visualize.mnist_transfer_purify \
        --source-sweep outputs/mnist_full_r12/at/legendre/d3r20c64/seed_sweep_1606 \
        --target-sweep outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep_a001_1206

Notebook:
    from analysis.visualize.mnist_transfer_purify import transfer_purify_analysis
    df, fig_summary, fig_montage = transfer_purify_analysis(SOURCE_SWEEP, TARGET_SWEEP, ...)
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from analysis.utils import find_model_checkpoint, get_best_run, load_run_config
from src.analysis.purification import LikelihoodPurification
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine
from src.utils.evasion import ProjectedGradientDescent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG (defaults; override via CLI flags or function args) -------------------------
SOURCE_SWEEP = "outputs/mnist_full_r12/at/legendre/d3r20c64/seed_sweep_1606"        # attack source = AT
TARGET_SWEEP = "outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep_a001_1206"  # transfer target = a=0.01
EPS = 0.3                 # attack budget, range-relative fraction
RADIUS = 0.3              # purification radius, range-relative fraction
TARGET_PER_CLASS = 10     # collect at least this many transferring examples per class
ATTACK_NUM_STEPS = 40
PURIFY_NUM_STEPS = 20
EVAL_BATCH_SIZE = 128
MAX_ATTACK_SAMPLES: Optional[int] = None  # cap #clean inputs attacked (None = whole test set)
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "figures/mnist/transfer_purify"

CATEGORIES = ["both", "source_only", "target_only", "neither"]


# --- Pure helpers (unit-tested; no model / no torch state) ------------------------------
def _transfer_mask(
    src_clean_pred: np.ndarray,
    tgt_clean_pred: np.ndarray,
    src_adv_pred: np.ndarray,
    tgt_adv_pred: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Boolean mask: keep examples both models classify clean-correctly AND both misclassify
    adversarially (the attack fools both = transfers)."""
    both_clean_ok = (src_clean_pred == labels) & (tgt_clean_pred == labels)
    both_adv_wrong = (src_adv_pred != labels) & (tgt_adv_pred != labels)
    return both_clean_ok & both_adv_wrong


def _categorize(src_success: np.ndarray, tgt_success: np.ndarray) -> np.ndarray:
    """Map per-example purification successes to a category label array.

    both -> source & target recover; source_only / target_only -> exactly one; neither -> none.
    """
    src = np.asarray(src_success, dtype=bool)
    tgt = np.asarray(tgt_success, dtype=bool)
    out = np.empty(len(src), dtype=object)
    out[src & tgt] = "both"
    out[src & ~tgt] = "source_only"
    out[~src & tgt] = "target_only"
    out[~src & ~tgt] = "neither"
    return out


# --- Model / data loading ---------------------------------------------------------------
def _load_model(sweep_dir: str, device: str) -> Tuple[ConditionalBornMachine, object]:
    """Load the best (highest clean-acc) run of a seed sweep; return (cbm, run_cfg)."""
    sweep = Path(sweep_dir)
    # analysis CSV mirrors the outputs/ path: <root>/outputs/X -> <root>/analysis/outputs/X
    parts = list(sweep.parts)
    if "outputs" in parts:
        i = parts.index("outputs")
        csv = Path(*parts[:i], "analysis", *parts[i:]) / "evaluation_data.csv"
    else:
        csv = sweep / "evaluation_data.csv"
    if csv.exists():
        best = get_best_run(pd.read_csv(csv), "acc", minimize=False)
        run_dir = sweep / str(best["run_name"])
    else:
        run_dir = sweep / "0"  # no CSV: fall back to first run
        logger.warning(f"No evaluation_data.csv at {csv}; using {run_dir}")
    run_cfg = load_run_config(run_dir)
    cbm = ConditionalBornMachine.load(str(find_model_checkpoint(run_dir)), accumulate=True)
    cbm.to(device)
    cbm.eval()
    cbm.cache_log_Z()
    logger.info(f"Loaded {run_dir}")
    return cbm, run_cfg


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
def _classify(cbm: ConditionalBornMachine, x: torch.Tensor, device: str) -> torch.Tensor:
    """Top-1 class predictions (CPU long tensor)."""
    return cbm.class_probabilities(x.to(device)).argmax(dim=1).cpu()


# --- Main pipeline ----------------------------------------------------------------------
def transfer_purify_analysis(
    source_sweep: str = SOURCE_SWEEP,
    target_sweep: str = TARGET_SWEEP,
    eps: float = EPS,
    radius: float = RADIUS,
    target_per_class: int = TARGET_PER_CLASS,
    attack_num_steps: int = ATTACK_NUM_STEPS,
    purify_num_steps: int = PURIFY_NUM_STEPS,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_attack_samples: Optional[int] = MAX_ATTACK_SAMPLES,
    seed: int = SEED,
    device: str = DEVICE,
    save_dir: Optional[str] = SAVE_DIR,
) -> Tuple[pd.DataFrame, "object", "object"]:
    """Run the full transfer-attack + differential-purification analysis.

    Returns (records_df, summary_fig, montage_fig).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    src_cbm, src_cfg = _load_model(source_sweep, device)
    tgt_cbm, _ = _load_model(target_sweep, device)

    # Both models share the legendre embedding => identical input_range; build data once.
    loader = _test_loader(src_cfg, src_cbm, eval_batch_size)
    lo, hi = float(src_cbm.input_range[0]), float(src_cbm.input_range[1])
    range_size = hi - lo
    abs_eps, abs_radius = eps * range_size, radius * range_size
    logger.info(f"range=[{lo:.3f},{hi:.3f}] size={range_size:.3f} | "
                f"abs_eps={abs_eps:.4f} abs_radius={abs_radius:.4f}")

    attack = ProjectedGradientDescent(norm="inf", num_steps=attack_num_steps, random_start=True)

    # --- Stage 1+2: attack source, keep examples that fool both models -------------------
    kept_clean, kept_adv, kept_y = [], [], []
    kept_src_adv_pred, kept_tgt_adv_pred = [], []
    per_class: Dict[int, int] = {c: 0 for c in range(src_cbm.out_dim)}
    n_seen = 0

    for x, y in loader:
        if max_attack_samples is not None and n_seen >= max_attack_samples:
            break
        n_seen += len(x)
        x = x.to(device)
        y_np = y.numpy()

        src_clean = _classify(src_cbm, x, device).numpy()
        tgt_clean = _classify(tgt_cbm, x, device).numpy()
        x_adv = attack.generate(src_cbm, x, y.to(device), abs_eps, device)
        src_adv = _classify(src_cbm, x_adv, device).numpy()
        tgt_adv = _classify(tgt_cbm, x_adv, device).numpy()

        mask = _transfer_mask(src_clean, tgt_clean, src_adv, tgt_adv, y_np)
        if mask.any():
            idx = np.nonzero(mask)[0]
            kept_clean.append(x[idx].cpu())
            kept_adv.append(x_adv[idx].cpu())
            kept_y.append(y_np[idx])
            kept_src_adv_pred.append(src_adv[idx])
            kept_tgt_adv_pred.append(tgt_adv[idx])
            for c in y_np[idx]:
                per_class[int(c)] += 1

        if all(v >= target_per_class for v in per_class.values()):
            logger.info(f"Reached >= {target_per_class} transferring per class after "
                        f"{n_seen} attacked inputs.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not kept_y:
        raise RuntimeError("No transferring adversarial examples found. Try a stronger "
                           "attack (more steps / larger eps) or a different target model.")

    clean_imgs = torch.cat(kept_clean)
    adv_imgs = torch.cat(kept_adv)
    labels = np.concatenate(kept_y)
    src_adv_pred = np.concatenate(kept_src_adv_pred)
    tgt_adv_pred = np.concatenate(kept_tgt_adv_pred)
    logger.info(f"Collected {len(labels)} transferring examples. "
                f"Per-class counts: {dict(sorted(per_class.items()))}")
    low = [c for c, v in per_class.items() if v < target_per_class]
    if low:
        logger.warning(f"Classes below target ({target_per_class}): {low}")

    # --- Stage 3: differential purification (self-purify, self-classify) -----------------
    purifier = LikelihoodPurification(norm="inf", num_steps=purify_num_steps, random_start=False)

    def _purify_all(cbm):
        """Purify all adv images (batched) with `cbm`; return (pred, logpx_adv, logpx_pur, pur_imgs)."""
        preds, lpx_adv, lpx_pur, pur_chunks = [], [], [], []
        for i in range(0, len(adv_imgs), eval_batch_size):
            xb = adv_imgs[i:i + eval_batch_size].to(device)
            with torch.no_grad():
                lpx_adv.append(cbm.marginal_log_probability(xb).cpu())
            xp, lpx = purifier.purify(cbm, xb, abs_radius, device)
            pred = _classify(cbm, xp, device)
            preds.append(pred); lpx_pur.append(lpx.cpu()); pur_chunks.append(xp.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return (torch.cat(preds).numpy(), torch.cat(lpx_adv).numpy(),
                torch.cat(lpx_pur).numpy(), torch.cat(pur_chunks))

    src_pur_pred, src_lpx_adv, src_lpx_pur, src_pur_imgs = _purify_all(src_cbm)
    tgt_pur_pred, tgt_lpx_adv, tgt_lpx_pur, tgt_pur_imgs = _purify_all(tgt_cbm)

    src_success = src_pur_pred == labels
    tgt_success = tgt_pur_pred == labels
    category = _categorize(src_success, tgt_success)

    records = pd.DataFrame({
        "true_label": labels,
        "src_pred_adv": src_adv_pred, "tgt_pred_adv": tgt_adv_pred,
        "src_pur_pred": src_pur_pred, "tgt_pur_pred": tgt_pur_pred,
        "src_success": src_success, "tgt_success": tgt_success,
        "src_logpx_adv": src_lpx_adv, "src_logpx_pur": src_lpx_pur,
        "tgt_logpx_adv": tgt_lpx_adv, "tgt_logpx_pur": tgt_lpx_pur,
        "category": category,
    })

    logger.info(f"Purification success -- source: {src_success.mean():.3f}, "
                f"target: {tgt_success.mean():.3f}")
    logger.info("Category counts: " + ", ".join(
        f"{c}={int((category == c).sum())}" for c in CATEGORIES))

    # --- Figures -------------------------------------------------------------------------
    summary_fig = _plot_summary(records, source_sweep, target_sweep)
    montage_fig = _plot_montage(records, clean_imgs, adv_imgs, src_pur_imgs, tgt_pur_imgs,
                                lo, hi)

    if save_dir is not None:
        out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
        records.to_csv(out / "transfer_purify_records.csv", index=False)
        summary_fig.savefig(out / "transfer_purify_summary.png", dpi=150, bbox_inches="tight")
        montage_fig.savefig(out / "transfer_purify_montage.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved CSV + 2 figures to {out}")

    return records, summary_fig, montage_fig


# --- Plotting ---------------------------------------------------------------------------
def _plot_summary(records: pd.DataFrame, source_sweep: str, target_sweep: str):
    import matplotlib.pyplot as plt

    LABEL_FS, TICK_FS = 14, 12
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) 2x2 contingency of purification success (source vs target).
    cont = np.zeros((2, 2), dtype=int)  # rows: src fail/ok, cols: tgt fail/ok
    for s, t in zip(records["src_success"], records["tgt_success"]):
        cont[int(s), int(t)] += 1
    axc.imshow(cont, cmap="Blues")
    axc.set_xticks([0, 1]); axc.set_xticklabels(["tgt fail", "tgt ok"], fontsize=TICK_FS)
    axc.set_yticks([0, 1]); axc.set_yticklabels(["src fail", "src ok"], fontsize=TICK_FS)
    axc.set_title("Purification success contingency", fontsize=LABEL_FS)
    for i in range(2):
        for j in range(2):
            axc.text(j, i, str(cont[i, j]), ha="center", va="center",
                     color="black", fontsize=LABEL_FS + 2)

    # (b) per-class purification-success rate, source vs target.
    classes = sorted(records["true_label"].unique())
    src_rate = [records.loc[records.true_label == c, "src_success"].mean() for c in classes]
    tgt_rate = [records.loc[records.true_label == c, "tgt_success"].mean() for c in classes]
    x = np.arange(len(classes)); w = 0.4
    axb.bar(x - w / 2, src_rate, w, label="source (AT)", color="#FF9800")
    axb.bar(x + w / 2, tgt_rate, w, label="target (gen.)", color="#4CAF50")
    axb.set_xticks(x); axb.set_xticklabels(classes, fontsize=TICK_FS)
    axb.set_xlabel("digit class", fontsize=LABEL_FS)
    axb.set_ylabel("purification success rate", fontsize=LABEL_FS)
    axb.set_ylim(0, 1.05); axb.grid(axis="y", alpha=0.3, ls="--"); axb.set_axisbelow(True)
    axb.legend(fontsize=TICK_FS)
    axb.set_title(f"n={len(records)} transferring examples", fontsize=LABEL_FS)

    fig.tight_layout()
    return fig


def _plot_montage(records, clean, adv, src_pur, tgt_pur, lo, hi, per_cat: int = 3):
    """Grid: columns [clean | adv | source-purified | target-purified], rows grouped by
    outcome category. Purified-cell title coloured by classification correctness."""
    import matplotlib.pyplot as plt

    D = clean.shape[1]
    img_dim = math.isqrt(D)

    def _img(t):
        return ((t.float() - lo) / (hi - lo)).reshape(img_dim, img_dim).numpy()

    # Pick up to `per_cat` representative rows per category (prefer distinct classes).
    rows: List[Tuple[int, str]] = []
    for cat in CATEGORIES:
        idx = records.index[records["category"] == cat].tolist()
        rows += [(i, cat) for i in idx[:per_cat]]
    if not rows:
        rows = [(records.index[0], records.loc[records.index[0], "category"])]

    col_titles = ["clean", "adv", "src purified", "tgt purified"]
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(7, 1.9 * n))
    axes = np.atleast_2d(axes)
    for r, (i, cat) in enumerate(rows):
        rec = records.loc[i]
        y = int(rec["true_label"])
        panels = [
            (_img(clean[i]), f"y={y}", "black"),
            (_img(adv[i]), f"adv->{int(rec['src_pred_adv'])}/{int(rec['tgt_pred_adv'])}", "black"),
            (_img(src_pur[i]), f"src->{int(rec['src_pur_pred'])}",
             "green" if rec["src_success"] else "red"),
            (_img(tgt_pur[i]), f"tgt->{int(rec['tgt_pur_pred'])}",
             "green" if rec["tgt_success"] else "red"),
        ]
        for c, (img, title, color) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8, color=color)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(cat, fontsize=9, rotation=90, labelpad=2)
    for c, t in enumerate(col_titles):
        axes[0, c].annotate(t, xy=(0.5, 1.35), xycoords="axes fraction",
                            ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-sweep", default=SOURCE_SWEEP)
    p.add_argument("--target-sweep", default=TARGET_SWEEP)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--radius", type=float, default=RADIUS)
    p.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    p.add_argument("--attack-num-steps", type=int, default=ATTACK_NUM_STEPS)
    p.add_argument("--purify-num-steps", type=int, default=PURIFY_NUM_STEPS)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--max-attack-samples", type=int, default=MAX_ATTACK_SAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--save-dir", default=SAVE_DIR)
    a = p.parse_args()

    transfer_purify_analysis(
        source_sweep=a.source_sweep, target_sweep=a.target_sweep, eps=a.eps, radius=a.radius,
        target_per_class=a.target_per_class, attack_num_steps=a.attack_num_steps,
        purify_num_steps=a.purify_num_steps, eval_batch_size=a.eval_batch_size,
        max_attack_samples=a.max_attack_samples, seed=a.seed, device=a.device,
        save_dir=a.save_dir,
    )
