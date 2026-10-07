"""Figures that need trained models, not only results: they load one seed's
checkpoint per model (the study's first seed by default, the one ``prune
--keep-one`` keeps) and reach it through the model interface (D69), so the MPS and
JEM go through the same code.

    density    2-D data: decision boundary and p(x) over the input square, one
               column per model
    samples    class-conditional samples (MPS: exact sampling; JEM: SGLD from
               noise): per-class means for images, the points for 2-D data
    transfer   PGD examples crafted on one model and passed to all: original,
               adversarial, and each model's likelihood-purified version, on
               examples that fool every model; plus a statistics file
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from bm4tc.analysis.purification import LikelihoodPurification
from bm4tc.core.attacks import ProjectedGradientDescent
from bm4tc.core.embeddings import range_size_of, rel_to_abs
from bm4tc.core.interface import class_probabilities
from bm4tc.core.jem.model import JEMMLP
from bm4tc.core.jem.sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from bm4tc.core.objective import set_seed
from bm4tc.pipeline import analyse
from bm4tc.pipeline.figures.plots import STYLE, plt, save
from bm4tc.pipeline.figures.results import Model, model_list
from bm4tc.pipeline.runs import Study, find_runs

SEED = 0


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _same(a, b) -> bool:
    if isinstance(b, (int, float)) and not isinstance(b, bool) and a is not None:
        return math.isclose(float(a), float(b), abs_tol=1e-12)
    return a == b


def run_dir(model: Model, seed: Optional[int] = None) -> Path:
    """The run of the model's cell with that seed (default: the study's first)."""
    seed = Study(model.study).seeds()[0] if seed is None else seed
    found = [path for path, manifest in find_runs(study=model.study, seed=seed)
             if all(_same(manifest["identity"].get(k), v) for k, v in model.where.items())]
    if len(found) != 1:
        raise LookupError(f"{model.study} {model.where} seed {seed}: {len(found)} finished "
                          f"runs, need one")
    if not (found[0] / "models").exists():
        raise FileNotFoundError(f"{found[0]}: checkpoint pruned; `train --replace` it "
                                f"(`prune --keep-one` keeps the first seed's)")
    return found[0]


def load(model: Model, seed: Optional[int] = None, batch_size: int = 256):
    """The model (eval mode, on the device), its data handler and its run config."""
    path = run_dir(model, seed)
    net, data = analyse.load(path, batch_size, device())
    net.eval()
    net.reset()
    return net, data, json.loads((path / "run.json").read_text())["config"]


def _batched(fn, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([fn(x[i:i + batch_size].to(device())).cpu()
                          for i in range(0, len(x), batch_size)])


def _to_unit(x: torch.Tensor, net) -> torch.Tensor:
    lo, hi = net.input_range
    return (x - lo) / (hi - lo)


def _from_unit(x: torch.Tensor, net) -> torch.Tensor:
    lo, hi = net.input_range
    return lo + x * (hi - lo)


def _image(x: torch.Tensor) -> np.ndarray:
    side = math.isqrt(x.shape[-1])
    if side * side != x.shape[-1]:
        raise ValueError(f"{x.shape[-1]} features are not a square image")
    return x.reshape(side, side).float().numpy()


# ── density ─────────────────────────────────────────────────────────────────

def density(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models``, optional ``resolution`` (150), ``seed``, ``show_data`` (true)."""
    from matplotlib.colors import LinearSegmentedColormap

    boundary = LinearSegmentedColormap.from_list("boundary", ["C0", "white", "C1"])
    res = spec.get("resolution", 150)
    models = model_list(spec["models"], spec.get("where"))
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(2, len(models), figsize=(2.4 * len(models), 4.8),
                                 squeeze=False)
        for col, model in enumerate(models):
            net, data, _ = load(model, spec.get("seed"))
            if data.data["train"].shape[1] != 2:
                raise ValueError(f"density needs 2-D data; {model.study} is not")
            lo, hi = net.input_range
            axis = torch.linspace(lo, hi, res)
            g1, g2 = torch.meshgrid(axis, axis, indexing="xy")
            grid = torch.stack([g1.ravel(), g2.ravel()], dim=1)
            p1 = _batched(lambda x: class_probabilities(net, x)[:, 1], grid).reshape(res, res)
            log_joint = _batched(lambda x: net.log_joint(x) - net.log_normalizer(), grid)
            joint = (log_joint - log_joint.max(dim=0).values).exp()   # each class's peak 1
            px = joint.sum(dim=1).reshape(res, res).numpy()

            top, bottom = axes[0, col], axes[1, col]
            top.pcolormesh(g1, g2, p1.numpy(), cmap=boundary, vmin=0, vmax=1, shading="auto")
            if spec.get("show_data", True):
                x, y = data.data["train"].cpu(), data.labels["train"].cpu()
                for c in range(net.out_dim):
                    top.scatter(*x[y == c].T, s=6, facecolors="white", edgecolors=f"C{c}",
                                linewidths=0.5, alpha=0.4)
            bottom.pcolormesh(g1, g2, px, cmap="Purples", shading="auto",
                              vmin=np.percentile(px, 2), vmax=np.percentile(px, 98))
            top.set_title(model.label)
            for ax in (top, bottom):
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
        axes[0, 0].set_ylabel(r"$p(c\,|\,x)$")
        axes[1, 0].set_ylabel(r"$p(x)$")
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        return save(fig, out)


# ── samples ─────────────────────────────────────────────────────────────────

def class_samples(net, config: Mapping, n_per_class: int, num_bins: int,
                  sgld_steps: int) -> torch.Tensor:
    """(classes, n, features) in the model's input range."""
    if isinstance(net, JEMMLP):
        sgld = SGLDConfig(**config["jem"]["sampler"])
        sampler = SGLDSampler(sgld, ReplayBuffer(1, net.data_dim, net.input_range))
        return torch.stack([sampler.sample_fresh(net, n_per_class, device(), class_idx=c,
                                                 num_steps=sgld_steps)
                            for c in range(net.out_dim)])
    net.renormalize_(log_target=0.0)
    x, y = net.sample_all_classes(n_per_class=n_per_class, num_bins=num_bins)
    return torch.stack([x[y == c] for c in range(net.out_dim)]).cpu()


def samples(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models``, optional ``n_per_class`` (64), ``num_bins`` (100, MPS),
    ``sgld_steps`` (1000, JEM), ``seed``. Images: one row per model, the mean
    sample of each class; 2-D data: one panel per model, the samples themselves."""
    models = model_list(spec["models"], spec.get("where"))
    drawn = []
    for model in models:
        net, _, config = load(model, spec.get("seed"))
        set_seed(SEED)
        x = class_samples(net, config, spec.get("n_per_class", 64),
                          spec.get("num_bins", 100), spec.get("sgld_steps", 1000))
        drawn.append((model, _to_unit(x, net).clamp(0, 1)))
    with plt.rc_context({**STYLE, "axes.grid": False}):
        if drawn[0][1].shape[-1] == 2:
            fig, axes = plt.subplots(1, len(drawn), squeeze=False,
                                     figsize=(2.4 * len(drawn), 2.4))
            for ax, (model, x) in zip(axes[0], drawn):
                for c, points in enumerate(x):
                    ax.scatter(*points.T, s=4, color=f"C{c}", alpha=0.6)
                ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal",
                       title=model.label)
            return save(fig, out)
        n_classes = drawn[0][1].shape[0]
        fig, axes = plt.subplots(len(drawn), n_classes, squeeze=False,
                                 figsize=(0.9 * n_classes, 1.0 * len(drawn)))
        for row, (model, x) in enumerate(drawn):
            for c, mean in enumerate(x.mean(dim=1)):
                ax = axes[row, c]
                ax.imshow(_image(mean), cmap="gray_r", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(str(c))
            axes[row, 0].set_ylabel(model.label, rotation=0, ha="right", va="center")
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        return save(fig, out)


# ── transfer ────────────────────────────────────────────────────────────────

def transfer_mask(clean: List[np.ndarray], adv: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Examples every model classifies correctly when clean and wrongly when attacked."""
    mask = np.ones(len(labels), dtype=bool)
    for c, a in zip(clean, adv):
        mask &= (c == labels) & (a != labels)
    return mask


def first_of_each(labels: np.ndarray, classes) -> Dict[int, Optional[int]]:
    """The first index of each class (test-set order, so not cherry-picked)."""
    return {int(c): (int(np.flatnonzero(labels == c)[0]) if (labels == c).any() else None)
            for c in classes}


def transfer_examples(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The computation behind ``transfer``, all images in [0, 1]: ``clean``,
    ``adv`` (N, features), ``labels`` (N,); ``keep`` (indices fooling every model);
    per model ``clean_pred``, ``adv_pred`` (N,), ``purified`` (len(keep), features),
    ``purified_pred``; ``models``; ``source`` (index)."""
    specs = list(spec["models"])
    sources = [i for i, s in enumerate(specs) if s.get("source")]
    if len(sources) != 1:
        raise ValueError("transfer: exactly one model needs `source: true`")
    models = model_list(specs, spec.get("where"))
    loaded = [load(m, spec.get("seed")) for m in models]
    nets = [n for n, _, _ in loaded]
    src = nets[sources[0]]
    data = loaded[sources[0]][1]
    n = spec.get("max_attack", 2000)
    x_all, y_all = data.data["test"][:n].cpu(), data.labels["test"][:n].cpu()
    eps, delta, batch = float(spec["eps"]), float(spec["delta"]), spec.get("batch_size", 128)
    attack = ProjectedGradientDescent(norm="inf", num_steps=spec.get("attack_steps", 40),
                                      random_start=True)
    purifier = LikelihoodPurification(norm="inf", num_steps=spec.get("purify_steps", 20))

    def predict(net, unit):
        return _batched(lambda x: class_probabilities(net, x).argmax(1),
                        _from_unit(unit, net)).numpy()

    set_seed(SEED)
    advs = []
    for i in range(0, len(x_all), batch):
        x, y = x_all[i:i + batch].to(device()), y_all[i:i + batch].to(device())
        advs.append(attack.generate(src, x, y, rel_to_abs(eps, range_size_of(src)),
                                    device()).detach().cpu())
    r = {"models": models, "source": sources[0], "labels": y_all.numpy(),
         "clean": _to_unit(x_all, src), "adv": _to_unit(torch.cat(advs), src)}
    r["clean_pred"] = [predict(net, r["clean"]) for net in nets]
    r["adv_pred"] = [predict(net, r["adv"]) for net in nets]
    r["keep"] = np.flatnonzero(transfer_mask(r["clean_pred"], r["adv_pred"], r["labels"]))
    if not len(r["keep"]):
        raise RuntimeError("transfer: no attack fools every model; raise eps or max_attack")
    r["purified"], r["purified_pred"] = [], []
    for net in nets:
        x = _from_unit(r["adv"][r["keep"]], net)
        chunks = []
        for i in range(0, len(x), batch):
            xp, _ = purifier.purify(net, x[i:i + batch].to(device()),
                                    rel_to_abs(delta, range_size_of(net)), device())
            chunks.append(xp.detach().cpu())
        r["purified"].append(_to_unit(torch.cat(chunks), net))
        r["purified_pred"].append(predict(net, r["purified"][-1]))
    return r


def transfer(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models`` (one with ``source: true``: the attacked one), ``eps``, ``delta``,
    optional ``classes`` ([0, 3, 5, 9]), ``attack_steps`` (40), ``purify_steps``
    (20), ``max_attack`` (2000 test examples), ``seed``. Budgets are relative, so
    models with different input ranges get the same attack in pixel terms. Writes
    the image grid and a ``.txt`` with clean / attacked / purified accuracy."""
    r = transfer_examples(spec)
    models, src, keep, labels = r["models"], r["source"], r["keep"], r["labels"]
    kept = labels[keep]
    rows = first_of_each(kept, spec.get("classes", [0, 3, 5, 9]))
    titles = ["original", "adversarial"] + [m.label for m in models]
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(len(rows), len(titles), squeeze=False,
                                 figsize=(1.3 * len(titles), 1.45 * len(rows)))
        for row, (c, i) in enumerate(rows.items()):
            panels = [] if i is None else (
                [(r["clean"][keep[i]], c), (r["adv"][keep[i]], r["adv_pred"][src][keep[i]])]
                + [(r["purified"][k][i], r["purified_pred"][k][i]) for k in range(len(models))])
            for j, ax in enumerate(axes[row]):
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[j], fontsize=8)
                if j < len(panels):
                    img, pred = panels[j]
                    ax.imshow(_image(img.clamp(0, 1)), cmap="gray_r", vmin=0, vmax=1)
                    ax.set_xlabel(f"pred. {int(pred)}", fontsize=7, labelpad=1)
            axes[row, 0].set_ylabel(f"class {c}", rotation=0, ha="right", va="center")
        fig.subplots_adjust(wspace=0.05, hspace=0.35)
        written = save(fig, out)

    lines = [f"PGD-{spec.get('attack_steps', 40)} eps_rel={float(spec['eps']):g} on "
             f"{models[src].label}; likelihood purification delta_rel={float(spec['delta']):g}; "
             f"{len(labels)} test examples attacked, {len(keep)} fool every model",
             f"{'model':<40} {'clean':>7} {'attacked':>9} {'purified':>9}"]
    for k, m in enumerate(models):
        lines.append(f"{m.label:<40} {np.mean(r['clean_pred'][k] == labels):7.3f} "
                     f"{np.mean(r['adv_pred'][k] == labels):9.3f} "
                     f"{np.mean(r['purified_pred'][k] == kept):9.3f}")
    stats = out.parent / f"{out.name}.txt"
    stats.write_text("\n".join(lines) + "\n")
    return written + [stats]


KINDS: Dict[str, Any] = {"density": density, "samples": samples, "transfer": transfer}
