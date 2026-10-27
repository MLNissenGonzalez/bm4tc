"""Post-hoc analysis of one finished run, on its test split (D20).

Each part loads nothing and writes nothing: it takes the model, the data and the
study's analysis settings and returns metric keys (:mod:`bm4tc.pipeline.metrics`).
Every part starts from the same seed, so its numbers do not depend on which
parts ran before it (a resumed analysis reproduces them).

    clean      acc, losses, the robust-accuracy ceilings (test; data law if known)
    uq         PGD at every budget: rob, detection, likelihood purification
    uq_joint   the same under the joint attack (if analysis.joint_attack)
    gradients  the classifier's first-order steepness at the attacked points (D169)
    gibbs     MPS: Gibbs purification of PGD examples   } if analysis.sweep_purify
    sgld       JEM: SGLD purification of PGD examples    } .enabled (D72)
    adaptive   the adaptive attack against likelihood purification (D96; if
               analysis.adaptive.enabled, on the runs of analysis.adaptive.seeds, D129)
    adaptive_det  the adaptive attack against the detector, also started on the
               data (D156; on the same runs as adaptive)
    adaptive_sweep  the adaptive attack against Gibbs / SGLD (D132; if both are
               enabled, on the runs of analysis.sweep_purify.adaptive_seeds)

PGD is one attack everywhere (D149): :class:`BarePGD`, ``analysis.attack.steps`` and
``.restarts`` on the fixed test subsample ``analysis.attack.subsample`` (the sweep
purifiers on ``sweep_purify.subsample``, a prefix of it). Its examples give ``rob``
and every transfer number, and start the adaptive attacks (D98), which run with
the same steps and restarts. The attack on the detector also starts one restart on
the data (:class:`DataStarts`, D156).

The model is the run's (``model: mps | jem``); every part but the sweep
purification is the same code for both (D35, D69). On CUDA, an MPS's per-batch
work (forwards, PGD, likelihood purification, the adaptive attack's steps)
replays from CUDA graphs, one set per part (D92, D117); JEM's stays eager, and so
does the sweep purification (Gibbs, SGLD) with its adaptive attack.
"""
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from bm4tc.pipeline.metrics import key
from bm4tc.analysis.ceiling import population_ceiling, robust_accuracy_ceiling
from bm4tc.analysis.adaptive import AdaptivePGD, BarePGD, DataStarts, Defense
from bm4tc.analysis.gradients import margin_sensitivity
from bm4tc.analysis.purification import GibbsPurification, LikelihoodPurification
from bm4tc.analysis.uq import (UQConfig, UQEvaluation, UQResults, compute_thresholds,
                               fixed_subsample)
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
    datahandler.split_and_rescale(model.input_range, seed=manifest.config.tracking.seed)
    datahandler.get_classification_loaders(batch_size=batch_size)
    return model, datahandler


# The analysis settings each part's numbers depend on (besides the budgets): a
# part is recomputed when one of them, the budgets or the run changes.
SETTINGS = {
    "clean": ("rob_ceiling", "rob_ceiling_cells"),
    "uq": ("attack", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "uq_joint": ("attack", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "gradients": ("attack", "batch_size"),
    "gibbs": ("attack", "batch_size", "sweep_purify"),
    "sgld": ("attack", "batch_size", "sweep_purify"),
    # D156 moved the detector to adaptive_det; adaptive keeps the whole adaptive section
    # in its hash (percentile included), so purification results from before stay valid.
    "adaptive": ("attack", "purify_delta", "purify_steps", "batch_size", "adaptive"),
    "adaptive_det": ("attack", "batch_size", "adaptive"),
    "adaptive_sweep": ("attack", "batch_size", "sweep_purify", "adaptive"),
}


# The parts that report log-densities (loss_x, log_px), and the units they report them in
# (D170). The units enter these parts' hashes, so results stored in the model's units
# (before D170) are recomputed unless converted in place first.
LOG_DENSITY_PARTS = ("clean", "uq", "uq_joint")
LOG_DENSITY_UNITS = "nats per feature on the unit interval"


def log_width(model) -> float:
    """ln(hi − lo) of the model's input range: a log-density per feature on that range
    minus this is the same density on the unit interval, the data's own range before
    rescaling (D170). So log-densities compare across embeddings (ranges of width 1, 2,
    1.98 and 8) and with JEM ([−1, 1]); the training logs (``loss_x`` on train and valid,
    the objective) stay on the model's range, where they are optimised."""
    return math.log(range_size_of(model))


def part_hash(name: str, analysis, budgets: List[float], run_hash: str) -> str:
    """What makes a part's results reusable: its settings, the budgets, the run.
    ``adaptive.seeds`` and ``sweep_purify.adaptive_seeds`` say which runs get a part,
    not how, so they are left out. ``sweep_purify.adaptive_restarts`` enters only the
    sweep purifier's attack, as the restarts it runs (D135); ``attack.subsample``
    does not enter the sweep parts, which attack ``sweep_purify.subsample``."""
    settings = {k: OmegaConf.to_container(analysis, resolve=True)[k] for k in SETTINGS[name]}
    for section, picks in (("adaptive", "seeds"), ("sweep_purify", "adaptive_seeds"),
                           ("sweep_purify", "adaptive_restarts")):
        if section in settings:
            settings[section].pop(picks, None)
    if name in ("gibbs", "sgld", "adaptive_sweep"):
        settings["attack"].pop("subsample")
    if name == "adaptive_sweep":
        settings["attack"]["restarts"] = _sweep_restarts(analysis)
    fields = {"part": name, "settings": settings, "budgets": list(budgets),
              "run": run_hash, "seed": SEED}
    if name in LOG_DENSITY_PARTS:
        fields["units"] = LOG_DENSITY_UNITS
    blob = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def parts(analysis, model: str = "mps", seed: Optional[int] = None) -> Dict[str, Callable]:
    """The parts the settings ask for, in order, for the run of ``seed``: the
    adaptive attacks on likelihood purification and the detector only on
    ``analysis.adaptive.seeds`` (D129), on the sweep purifier only on
    ``analysis.sweep_purify.adaptive_seeds`` (D132); null is every seed.

    A run's metrics merge the parts' results in this order, later parts winning
    (``stages.analysed``): ``adaptive`` results stored before D156 still hold the
    detector's keys, which ``adaptive_det`` replaces."""
    out = {"clean": clean, "uq": uq}
    if analysis.joint_attack:
        out["uq_joint"] = uq_joint
    out["gradients"] = gradients
    if analysis.sweep_purify.enabled:
        out.update({"sgld": sgld} if model == "jem" else {"gibbs": gibbs})
    def picked(seeds):
        return seeds is None or seed in seeds

    if analysis.adaptive.enabled and picked(analysis.adaptive.seeds):
        out["adaptive"] = adaptive
        out["adaptive_det"] = adaptive_det
    if (analysis.adaptive.enabled and analysis.sweep_purify.enabled
            and picked(analysis.sweep_purify.adaptive_seeds)):
        out["adaptive_sweep"] = adaptive_sweep
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
    if not jem:
        out[key("loss_x", SPLIT)] -= log_width(cbm)
    if analysis.rob_ceiling:
        out.update(_ceilings(cbm, datahandler, budgets, analysis.rob_ceiling_cells))
    return out


def _ceilings(cbm, datahandler, budgets, cells: int) -> Dict[str, float]:
    """Bounds on robust accuracy without (``rob_ceiling``) and with rejection
    (``rob_ceiling_sel``): on the test split (two classes), and on the data law
    under the pseudo-split ``pop`` where the law is known (spirals), the upper end
    of the bracket (D97, notes:adaptive §2)."""
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
        norm="inf", eps_rel=list(budgets), attack_num_steps=analysis.attack.steps,
        attack_restarts=analysis.attack.restarts, attack_subsample=analysis.attack.subsample,
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


def _attack_keys(r: UQResults, n_features: int, log_w: float,
                 suffix: str = "") -> Dict[str, float]:
    """``log_px`` is per feature (D86) on the unit interval (D170), like ``loss_x``;
    ``log_w`` is :func:`log_width` of the model."""
    out = {}
    for eps, acc in r.adv_accuracies.items():
        out[key(f"rob{suffix}", SPLIT, eps)] = acc
        out[key(f"log_px{suffix}", SPLIT, eps)] = (float(r.adv_log_px[eps].mean()) / n_features
                                                   + log_w)
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
    out = _attack_keys(r, n, log_width(cbm))
    out[key("log_px", SPLIT, 0)] = float(r.clean_log_px.mean()) / n + log_width(cbm)
    for pct, rate in r.clean_flagged.items():
        out[key("detect", SPLIT, 0, f"q{pct:g}")] = rate
    for delta, m in r.clean_purification_results.items():
        out[key("purify", SPLIT, 0, f"d{delta:g}")] = m.accuracy_after_purify
    return out


def gradients(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    """The classifier's first-order steepness (D169, :mod:`bm4tc.analysis.gradients`) on
    the attack's fixed points, as medians over them, budgets relative to the range:
    the margin, the gain G per unit of budget, the share of G from features at the
    range's ends, the flip budget ε* (0 for a point already wrong), and per budget the
    fraction of points with ε* below it (a first-order estimate of 1 − rob)."""
    cbm.to(device)
    X, y = datahandler.data[SPLIT], datahandler.labels[SPLIT]
    idx = fixed_subsample(len(X), analysis.attack.subsample)
    s = margin_sensitivity(cbm, X[idx], y[idx], analysis.batch_size, device)
    gain = s.gain * range_size_of(cbm)
    eps_star = s.margin.clamp(min=0) / gain
    out = {key("margin", SPLIT): float(s.margin.median()),
           key("gain", SPLIT): float(gain.median()),
           key("gain_ends", SPLIT): float((s.gain_ends / s.gain).nan_to_num().median()),
           key("eps_star", SPLIT): float(eps_star.median())}
    for eps in budgets:
        out[key("eps_star_below", SPLIT, eps)] = float((eps_star < eps).float().mean())
    return out


def uq_joint(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    cfg = _uq_config(analysis, budgets, attack_method="JOINT_PGD")
    return _attack_keys(_evaluate(cbm, datahandler, cfg, device), _n_features(datahandler),
                        log_width(cbm), "_joint")


def _sweep(name: str, cbm, datahandler, analysis, budgets, device, purifier=None):
    """Sweep purification of the PGD examples and of clean data, keys
    ``purify_{name}`` / ``recovery_{name}`` per sweep count."""
    p = analysis.sweep_purify
    cfg = _uq_config(analysis, budgets, run_sweeps=True, sweeps=list(p.sweeps),
                     gibbs_num_bins=p.gibbs.num_bins, gibbs_batch_size=p.gibbs.batch_size,
                     gibbs_step_delta_rel=p.step, sweep_subsample=p.subsample)
    cfg.percentiles, cfg.delta_rel = [], []   # only sweeps: no detection, no likelihood purification
    cfg.attack_subsample = p.subsample        # PGD only on the swept points
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


def _likelihood_purified(purifier, delta_abs, x):
    return purifier(x, delta_abs)[0]


def _sweep_purified(purifier, model, sweeps, device, x):
    return purifier.purify_snapshots(model, x, [sweeps], device)[sweeps][0]


def _likelihood_defences(cbm, datahandler, analysis, graphs: Graphs, device):
    """(key name, setting, Defense, selective, test subsample) for likelihood
    purification at each radius; the purifier replays from ``graphs``."""
    range_size = range_size_of(cbm)
    likelihood = LikelihoodPurification(norm="inf", num_steps=analysis.purify_steps
                                        ).graphed(cbm, graphs)
    size = analysis.attack.subsample
    out = []
    for delta in analysis.purify_delta:
        purify = partial(_likelihood_purified, likelihood,
                         graphs.scalar(rel_to_abs(delta, range_size), device))
        out.append(("purify", f"d{delta:g}", Defense(cbm, purify), False, size))
    return out


def _detector_defences(cbm, datahandler, analysis, graphs: Graphs, device):
    """The detector at its threshold, as :func:`_likelihood_defences` (D122: one
    threshold; the detector on the purified input is not attacked, it matched
    purification alone)."""
    thresholds, _ = compute_thresholds(cbm, datahandler.classification["valid"],
                                       [analysis.adaptive.percentile], device)
    return [("rob_sel", f"q{q:g}", Defense(cbm, threshold=tau), True,
             analysis.attack.subsample) for q, tau in thresholds.items()]


def _sweep_defences(cbm, datahandler, analysis, graphs: Graphs, device):
    """The sweep purifier (Gibbs or SGLD) at each sweep count, on its subsample."""
    p = analysis.sweep_purify
    name, purifier = _sweep_purifier(cbm, analysis, progress=False)
    return [(f"purify_{name}", f"k{k}",
             Defense(cbm, partial(_sweep_purified, purifier, cbm, k, device), random=True),
             False, p.subsample) for k in p.sweeps]


def adaptive(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    """Likelihood purification at each budget, attacked by the adaptive attack
    (``purify_adapt``, D96), which starts its first restart from the PGD example
    (D98); PGD on the purifier alone is the ``uq`` part's already.

    On the test subsample ``analysis.attack.subsample`` (D117), at the attack's steps
    and restarts (D149). On CUDA, PGD, the likelihood purifier and the adaptive
    attack's steps replay from the part's graphs."""
    return _attack(cbm, datahandler, analysis, budgets, device, _likelihood_defences)


def adaptive_det(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    """The detector at ``analysis.adaptive.percentile`` (D122: an attack per
    threshold, so one) at each budget: selective robust accuracy under the PGD
    examples (``rob_sel``) and under the adaptive attack (``rob_sel_adapt``), whose
    first restarts begin at the PGD example and at a start on the data
    (:class:`DataStarts`, D156). Its own part, so adding the data start re-ran only
    the detector (cheap) and not the purifier (≈ 20× the cost on MNIST).

    Subsample, steps, restarts and graphs as :func:`adaptive`."""
    return _attack(cbm, datahandler, analysis, budgets, device, _detector_defences)


def adaptive_sweep(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    """The sweep purifier (Gibbs, SGLD) at each sweep count and budget, attacked by
    the adaptive attack (``purify_{gibbs,sgld}_adapt``, D96) on
    ``analysis.sweep_purify.subsample``. Its own part (D132): it is eager (the
    sampler is Python per site) and costs far more than the others on long chains,
    so ``sweep_purify.adaptive_seeds`` limits it on its own, and
    ``sweep_purify.adaptive_restarts`` may give it fewer restarts (D135)."""
    return _attack(cbm, datahandler, analysis, budgets, device, _sweep_defences,
                   restarts=_sweep_restarts(analysis))


def _sweep_restarts(analysis) -> int:
    r = analysis.sweep_purify.adaptive_restarts
    return analysis.attack.restarts if r is None else r


def _attack(cbm, datahandler, analysis, budgets, device, defences_of,
            restarts: Optional[int] = None) -> Dict[str, float]:
    """The defences ``defences_of`` builds, attacked at each budget by the adaptive
    attack and, for a selective one, also by the PGD examples, both at
    ``analysis.attack``'s steps and ``restarts`` (default ``analysis.attack.restarts``).
    The adaptive attack starts at the PGD example and, on a selective defence, also
    on the data (D156). A random purifier's attack stays eager."""
    a, steps = analysis.adaptive, analysis.attack.steps
    restarts = analysis.attack.restarts if restarts is None else restarts
    cbm.to(device)
    cbm.log_normalizer()
    X, y = datahandler.data[SPLIT], datahandler.labels[SPLIT]
    range_size = range_size_of(cbm)
    graphs, eager = _graphs(cbm, device), Graphs(enabled=False)
    pgd = BarePGD(steps, restarts).graphed(cbm, graphs)
    attack = AdaptivePGD(steps=steps, restarts=restarts, eot=a.eot)
    defences = defences_of(cbm, datahandler, analysis, graphs, device)
    data_starts = [DataStarts(d[2], datahandler.data["train"], analysis.batch_size)
                   if d[3] else None for d in defences]
    # one graphed attack per defence, captured once for every budget
    attacks = [(attack.graphed(d[2], g), g)
               for d, g in ((d, eager if d[2].random else graphs) for d in defences)]
    # PGD on the points some defence is scored on (sorted); the other rows stay nan
    attacked = torch.unique(torch.cat([fixed_subsample(len(X), d[4]) for d in defences]))
    out = {}
    for eps in budgets:
        eps_abs = rel_to_abs(eps, range_size)
        x_pgd = torch.full_like(X, float("nan"))
        for i in range(0, len(attacked), analysis.batch_size):
            b = attacked[i:i + analysis.batch_size]
            x_pgd[b] = pgd(X[b].to(device), y[b].to(device),
                           graphs.scalar(eps_abs, device)).cpu()
        for (name, setting, defense, selective, subsample), (adapt, g), on_data in zip(
                defences, attacks, data_starts):
            idx = fixed_subsample(len(X), subsample)
            radius = g.scalar(eps_abs, device)

            def starts(xb, yb, b):
                return [x_pgd[b]] + ([] if on_data is None else [on_data(xb, yb)])

            crafts = {"_adapt": lambda xb, yb, b: adapt(xb, yb, radius, starts(xb, yb, b))}
            if selective:
                crafts[""] = None   # the PGD examples
            for suffix, craft in crafts.items():
                start = time.time()
                scores = []
                for i in range(0, len(idx), analysis.batch_size):
                    b = idx[i:i + analysis.batch_size]
                    xb, yb = X[b].to(device), y[b].to(device)
                    x_adv = x_pgd[b].to(device) if craft is None else craft(xb, yb, b)
                    scores.append(_score(defense, xb, x_adv, yb, selective, a.eval_draws).cpu())
                k = key(name + suffix, SPLIT, eps, setting)
                out[k] = float(torch.cat(scores).mean())
                logger.info(f"  {k} = {out[k]:.4f} ({time.time() - start:.0f} s)")
    return out


def _score(defense: Defense, x, x_adv, labels, selective: bool, draws: int) -> torch.Tensor:
    """Per sample: the probability of a right answer on ``x_adv``, or with
    ``selective`` of a right answer on ``x`` and no wrong one on ``x_adv``
    (notes:adaptive §1)."""
    right_adv, error_adv = defense.rates(x_adv, labels, draws)
    if not selective:
        return right_adv
    right_clean, _ = defense.rates(x, labels, draws)
    return right_clean * (1 - error_adv)


def run_part(part: Callable, cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    set_seed(SEED)
    return part(cbm, datahandler, analysis, budgets, device)
