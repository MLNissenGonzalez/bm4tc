"""Post-hoc analysis of one finished run, on its test split (D20).

Each part loads nothing and writes nothing: it takes the model, the data and the
study's analysis settings and returns metric keys (:mod:`bm4tc.pipeline.metrics`).
Every part starts from the same seed, so its numbers do not depend on which
parts ran before it (a resumed analysis reproduces them).

    clean      acc, losses, the robust-accuracy ceilings (test; data law if known)
    uq         PGD at every budget: rob, detection, likelihood purification
    uq_joint   the same under the joint attack (if analysis.joint_attack)
    gibbs      MPS: Gibbs purification of PGD examples   } if analysis.sweep_purify
    sgld       JEM: SGLD purification of PGD examples    } .enabled (D72)
    adaptive   PGD and the adaptive attack against purifier and detector (D96; if
               analysis.adaptive.enabled; eager)

The model is the run's (``model: mps | jem``); every part but the sweep
purification is the same code for both (D35, D69). On CUDA, an MPS's per-batch
work (forwards, PGD, likelihood purification) replays from CUDA graphs, one set
per part (D92); JEM's stays eager.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from bm4tc.pipeline.metrics import key
from bm4tc.analysis.ceiling import population_ceiling, robust_accuracy_ceiling
from bm4tc.analysis.adaptive import AdaptivePGD, Defense
from bm4tc.analysis.purification import GibbsPurification, LikelihoodPurification
from bm4tc.analysis.uq import UQConfig, UQEvaluation, UQResults, compute_thresholds
from bm4tc.core.attacks import ProjectedGradientDescent
from bm4tc.pipeline.data import DataHandler
from bm4tc.core.jem.model import JEMMLP
from bm4tc.core.jem.purification import SGLDPurification
from bm4tc.core.model import ConditionalBornMachine
from bm4tc.core.embeddings import range_size_of, rel_to_abs
from bm4tc.core.graphs import Graphs
from bm4tc.core.objective import Evaluation, set_seed

logger = logging.getLogger(__name__)

SEED = 0
SPLIT = "test"


def load(run_dir: Path, batch_size: int, device: torch.device):
    """The run's selected model and its data, split and rescaled as in training."""
    manifest = OmegaConf.create((Path(run_dir) / "run.json").read_text())
    checkpoint = str(Path(run_dir) / "models" / "model")
    if manifest.config.model == "jem":
        model, _ = JEMMLP.load(checkpoint, device)
    else:
        model = ConditionalBornMachine.load(checkpoint)
    model.to(device)
    datahandler = DataHandler(manifest.config.dataset)
    datahandler.load()
    datahandler.split_and_rescale(model.input_range)
    datahandler.get_classification_loaders(batch_size=batch_size)
    return model, datahandler


# The analysis settings each part's numbers depend on (besides the budgets): a
# part is recomputed when one of them, the budgets or the run changes.
SETTINGS = {
    "clean": ("rob_ceiling", "rob_ceiling_cells"),
    "uq": ("attack_steps", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "uq_joint": ("attack_steps", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "gibbs": ("attack_steps", "batch_size", "sweep_purify"),
    "sgld": ("attack_steps", "batch_size", "sweep_purify"),
    "adaptive": ("attack_steps", "percentiles", "purify_delta", "purify_steps", "batch_size",
                 "sweep_purify", "adaptive"),
}


def part_hash(name: str, analysis, budgets: List[float], run_hash: str) -> str:
    """What makes a part's results reusable: its settings, the budgets, the run."""
    settings = {k: OmegaConf.to_container(analysis, resolve=True)[k] for k in SETTINGS[name]}
    blob = json.dumps({"part": name, "settings": settings, "budgets": list(budgets),
                       "run": run_hash, "seed": SEED}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def parts(analysis, model: str = "mps") -> Dict[str, Callable]:
    """The parts the settings ask for, in order."""
    out = {"clean": clean, "uq": uq}
    if analysis.joint_attack:
        out["uq_joint"] = uq_joint
    if analysis.sweep_purify.enabled:
        out.update({"sgld": sgld} if model == "jem" else {"gibbs": gibbs})
    if analysis.adaptive.enabled:
        out["adaptive"] = adaptive
    return out


def _graphs(cbm, device) -> Graphs:
    """A part's CUDA graphs (D92): on CUDA, for the MPS. JEM stays eager: its
    capture has not been checked."""
    return Graphs(enabled=torch.device(device).type == "cuda"
                  and isinstance(cbm, ConditionalBornMachine))


def clean(cbm, datahandler, analysis, budgets: List[float], device) -> Dict[str, float]:
    jem = isinstance(cbm, JEMMLP)   # no exact log Z: no loss_x on test
    m = Evaluation(cbm, graphs=_graphs(cbm, device))(
        datahandler.classification[SPLIT], device, log_Z=float("nan") if jem else None)
    names = ("acc", "loss_dis") if jem else ("acc", "loss_dis", "loss_x")
    out = {key(name, SPLIT): m[name] for name in names}
    if analysis.rob_ceiling:
        out.update(_ceilings(cbm, datahandler, budgets, analysis.rob_ceiling_cells))
    return out


def _ceilings(cbm, datahandler, budgets, cells: int) -> Dict[str, float]:
    """Bounds on robust accuracy without (``rob_ceiling``) and with rejection
    (``rob_ceiling_sel``): on the test split (two classes), and on the data law
    under the pseudo-split ``pop`` where the law is known (spirals), the upper end
    of the bracket (D97, docs/adaptive.md §2)."""
    X = datahandler.data[SPLIT].detach().cpu().numpy()
    y = datahandler.labels[SPLIT].detach().cpu().numpy()
    edges = np.linspace(*cbm.input_range, cells + 1)
    masses = datahandler.class_masses(edges)
    out = {}
    for eps in budgets:
        eps_abs = rel_to_abs(eps, range_size_of(cbm))
        for name, selective in (("rob_ceiling", False), ("rob_ceiling_sel", True)):
            ceiling = robust_accuracy_ceiling(X, y, eps_abs, selective=selective)
            if ceiling == ceiling:  # nan: not two classes, or too many points
                out[key(name, SPLIT, eps)] = ceiling
            if masses is not None:
                _, upper = population_ceiling(masses[0], masses[1], edges[1] - edges[0],
                                              eps_abs, selective=selective)
                out[key(name, "pop", eps)] = upper
    return out


def _uq_config(analysis, budgets, **kw) -> UQConfig:
    return UQConfig(
        norm="inf", eps_rel=list(budgets), attack_num_steps=analysis.attack_steps,
        percentiles=list(analysis.percentiles), delta_rel=list(analysis.purify_delta),
        num_steps=analysis.purify_steps, eval_batch_size=analysis.batch_size, **kw)


def _evaluate(cbm, datahandler, cfg: UQConfig, device, sweep_purifier=None) -> UQResults:
    r = UQEvaluation(cfg).evaluate(
        cbm, datahandler.classification[SPLIT], device,
        calib_loader=datahandler.classification["valid"], sweep_purifier=sweep_purifier,
        graphs=_graphs(cbm, device))
    # UQEvaluation logs a failed budget and goes on; a part must not be kept with
    # a hole in it.
    expected = {
        "attacks": (len(r.adv_accuracies), len(cfg.eps_rel)),
        "purifications": (len(r.purification_results), len(cfg.eps_rel) * len(cfg.delta_rel)),
        "sweep purifications": (len(r.sweep_purification_results),
                                len(cfg.eps_rel) * len(cfg.sweeps) if cfg.run_sweeps else 0),
    }
    for what, (got, want) in expected.items():
        if got != want:
            raise RuntimeError(f"{want - got} of {want} {what} failed; see the log")
    return r


def _n_features(datahandler) -> int:
    return datahandler.data[SPLIT][0].numel()


def _attack_keys(r: UQResults, n_features: int, suffix: str = "") -> Dict[str, float]:
    """``log_px`` is per feature (D86), like ``loss_x``."""
    out = {}
    for eps, acc in r.adv_accuracies.items():
        out[key(f"rob{suffix}", SPLIT, eps)] = acc
        out[key(f"log_px{suffix}", SPLIT, eps)] = float(r.adv_log_px[eps].mean()) / n_features
    for (pct, eps), rate in r.detection_rates.items():
        q = f"q{pct:g}"
        out[key(f"detect{suffix}", SPLIT, eps, q)] = rate
        out[key(f"err_detected{suffix}", SPLIT, eps, q)] = r.err_rate_detected[(pct, eps)]
        out[key(f"err_passed{suffix}", SPLIT, eps, q)] = r.err_rate_passed[(pct, eps)]
    for (eps, delta), m in r.purification_results.items():
        d = f"d{delta:g}"
        out[key(f"purify{suffix}", SPLIT, eps, d)] = m.accuracy_after_purify
        out[key(f"recovery{suffix}", SPLIT, eps, d)] = m.recovery_rate
    return out


def uq(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    r = _evaluate(cbm, datahandler, _uq_config(analysis, budgets), device)
    n = _n_features(datahandler)
    out = _attack_keys(r, n)
    out[key("log_px", SPLIT, 0)] = float(r.clean_log_px.mean()) / n
    for pct, rate in r.clean_flagged.items():
        out[key("detect", SPLIT, 0, f"q{pct:g}")] = rate
    for delta, m in r.clean_purification_results.items():
        out[key("purify", SPLIT, 0, f"d{delta:g}")] = m.accuracy_after_purify
    return out


def uq_joint(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    cfg = _uq_config(analysis, budgets, attack_method="JOINT_PGD")
    return _attack_keys(_evaluate(cbm, datahandler, cfg, device), _n_features(datahandler),
                        "_joint")


def _sweep(name: str, cbm, datahandler, analysis, budgets, device, purifier=None):
    """Sweep purification of the PGD examples and of clean data, keys
    ``purify_{name}`` / ``recovery_{name}`` per sweep count."""
    p = analysis.sweep_purify
    cfg = _uq_config(analysis, budgets, run_sweeps=True, sweeps=list(p.sweeps),
                     gibbs_num_bins=p.gibbs.num_bins, gibbs_batch_size=p.gibbs.batch_size,
                     gibbs_step_delta_rel=p.step, sweep_subsample=p.subsample)
    cfg.percentiles, cfg.delta_rel = [], []   # only sweeps: no detection, no likelihood purification
    r = _evaluate(cbm, datahandler, cfg, device, sweep_purifier=purifier)
    out = {}
    for (eps, k), m in r.sweep_purification_results.items():
        out[key(f"purify_{name}", SPLIT, eps, f"k{k}")] = m.accuracy_after_purify
        out[key(f"recovery_{name}", SPLIT, eps, f"k{k}")] = m.recovery_rate
    for k, m in r.clean_sweep_purification_results.items():
        out[key(f"purify_{name}", SPLIT, 0, f"k{k}")] = m.accuracy_after_purify
    return out


def _sweep_purifier(cbm, analysis, progress: bool = True):
    """The model's sweep purifier: ``("sgld", ...)`` for JEM, ``("gibbs", ...)`` for the MPS."""
    p = analysis.sweep_purify
    if isinstance(cbm, JEMMLP):
        return "sgld", SGLDPurification(step_rel=p.step, steps=p.sgld.steps,
                                        step_size=p.sgld.step_size,
                                        noise_std=p.sgld.noise_std, batch_size=p.sgld.batch_size)
    return "gibbs", GibbsPurification(num_bins=p.gibbs.num_bins, gibbs_batch_size=p.gibbs.batch_size,
                                      step_delta_rel=p.step, progress=progress)


def gibbs(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    return _sweep("gibbs", cbm, datahandler, analysis, budgets, device,
                  _sweep_purifier(cbm, analysis)[1])


def sgld(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    return _sweep("sgld", cbm, datahandler, analysis, budgets, device,
                  _sweep_purifier(cbm, analysis)[1])


def _likelihood_purified(purifier, model, delta_abs, device, x):
    return purifier.purify(model, x, delta_abs, device)[0]


def _sweep_purified(purifier, model, sweeps, device, x):
    return purifier.purify_snapshots(model, x, [sweeps], device)[sweeps][0]


def _subsample(n: int, size: Optional[int]) -> torch.Tensor:
    """The fixed test subsample of the sweep purification (the same points as
    UQEvaluation's, seed 0)."""
    if size is None or size >= n:
        return torch.arange(n)
    return torch.from_numpy(np.random.default_rng(0).permutation(n)[:size])


def _defences(cbm, analysis, thresholds, range_size, device):
    """(key name, setting, Defense, selective, test subsample) for every defence:
    likelihood purification, the detector at each threshold, both, and the sweep
    purification if it is on."""
    likelihood = LikelihoodPurification(norm="inf", num_steps=analysis.purify_steps)
    out = []
    for delta in analysis.purify_delta:
        purify = partial(_likelihood_purified, likelihood, cbm, rel_to_abs(delta, range_size),
                         device)
        out.append(("purify", f"d{delta:g}", Defense(cbm, purify), False, None))
        for q, tau in thresholds.items():
            out.append(("purify_sel", f"d{delta:g}_q{q:g}",
                        Defense(cbm, purify, tau, detect_on="purified"), True, None))
    for q, tau in thresholds.items():
        out.append(("rob_sel", f"q{q:g}", Defense(cbm, threshold=tau), True, None))
    p = analysis.sweep_purify
    if p.enabled:
        name, purifier = _sweep_purifier(cbm, analysis, progress=False)
        for k in p.sweeps:
            out.append((f"purify_{name}", f"k{k}",
                        Defense(cbm, partial(_sweep_purified, purifier, cbm, k, device),
                                random=True), False, p.subsample))
    return out


def adaptive(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    """Each defence at each budget, attacked by PGD on the classifier (no suffix) and
    by the adaptive attack (``_adapt``, D96). Robust accuracy for the purifiers,
    selective robust accuracy (``rob_sel``, ``purify_sel``) where a detector may
    reject. PGD on the purifiers alone is the ``uq`` and sweep parts' already."""
    a = analysis.adaptive
    cbm.to(device)
    cbm.log_normalizer()
    thresholds, _ = compute_thresholds(cbm, datahandler.classification["valid"],
                                       list(analysis.percentiles), device)
    X, y = datahandler.data[SPLIT], datahandler.labels[SPLIT]
    range_size = range_size_of(cbm)
    pgd = ProjectedGradientDescent(norm="inf", num_steps=analysis.attack_steps)
    attack = AdaptivePGD(steps=a.steps, restarts=a.restarts, eot=a.eot)
    defences = _defences(cbm, analysis, thresholds, range_size, device)
    out = {}
    for eps in budgets:
        eps_abs = rel_to_abs(eps, range_size)
        x_pgd = torch.cat([pgd.generate(cbm, X[i:i + analysis.batch_size],
                                        y[i:i + analysis.batch_size], eps_abs, device).cpu()
                           for i in range(0, len(X), analysis.batch_size)])
        for name, setting, defense, selective, subsample in defences:
            idx = _subsample(len(X), subsample)
            crafts = {"_adapt": lambda xb, yb: attack.generate(defense, xb, yb, eps_abs)}
            if selective:
                crafts[""] = None   # the PGD examples
            for suffix, craft in crafts.items():
                start = time.time()
                scores = []
                for i in range(0, len(idx), analysis.batch_size):
                    b = idx[i:i + analysis.batch_size]
                    xb, yb = X[b].to(device), y[b].to(device)
                    x_adv = x_pgd[b].to(device) if craft is None else craft(xb, yb)
                    scores.append(_score(defense, xb, x_adv, yb, selective, a.eval_draws).cpu())
                k = key(name + suffix, SPLIT, eps, setting)
                out[k] = float(torch.cat(scores).mean())
                logger.info(f"  {k} = {out[k]:.4f} ({time.time() - start:.0f} s)")
    return out


def _score(defense: Defense, x, x_adv, labels, selective: bool, draws: int) -> torch.Tensor:
    """Per sample: the probability of a right answer on ``x_adv``, or with
    ``selective`` of a right answer on ``x`` and no wrong one on ``x_adv``
    (docs/adaptive.md §1)."""
    right_adv, error_adv = defense.rates(x_adv, labels, draws)
    if not selective:
        return right_adv
    right_clean, _ = defense.rates(x, labels, draws)
    return right_clean * (1 - error_adv)


def run_part(part: Callable, cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    set_seed(SEED)
    return part(cbm, datahandler, analysis, budgets, device)
