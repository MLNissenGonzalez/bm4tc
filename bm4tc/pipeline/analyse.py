"""Post-hoc analysis of one finished run, on its test split (D20).

Each part loads nothing and writes nothing: it takes the model, the data and the
study's analysis settings and returns metric keys (:mod:`bm4tc.pipeline.metrics`).
Every part starts from the same seed, so its numbers do not depend on which
parts ran before it (a resumed analysis reproduces them).

    clean      acc, losses, the robust-accuracy ceiling
    uq         PGD at every budget: rob, detection, likelihood purification
    uq_joint   the same under the joint attack (if analysis.joint_attack)
    gibbs      MPS: Gibbs purification of PGD examples   } if analysis.sweep_purify
    sgld       JEM: SGLD purification of PGD examples    } .enabled (D72)

The model is the run's (``model: mps | jem``); every part but the sweep
purification is the same code for both (D35, D69).
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List

import torch
from omegaconf import OmegaConf

from bm4tc.pipeline.metrics import key
from bm4tc.analysis.ceiling import robust_accuracy_ceiling
from bm4tc.analysis.uq import UQConfig, UQEvaluation, UQResults
from bm4tc.pipeline.data import DataHandler
from bm4tc.core.jem.model import JEMMLP
from bm4tc.core.jem.purification import SGLDPurification
from bm4tc.core.model import ConditionalBornMachine
from bm4tc.core.embeddings import range_size_of, rel_to_abs
from bm4tc.core.objective import evaluate, set_seed

SEED = 0
SPLIT = "test"


def load(run_dir: Path, batch_size: int, device: torch.device):
    """The run's selected model and its data, split and rescaled as in training."""
    manifest = OmegaConf.create((Path(run_dir) / "run.json").read_text())
    checkpoint = str(Path(run_dir) / "models" / "model")
    if manifest.config.model == "jem":
        model, _ = JEMMLP.load(checkpoint, device)
    else:
        model = ConditionalBornMachine.load(checkpoint, accumulate=True)
    model.to(device)
    datahandler = DataHandler(manifest.config.dataset)
    datahandler.load()
    datahandler.split_and_rescale(model.input_range)
    datahandler.get_classification_loaders(batch_size=batch_size)
    return model, datahandler


# The analysis settings each part's numbers depend on (besides the budgets): a
# part is recomputed when one of them, the budgets or the run changes.
SETTINGS = {
    "clean": ("rob_ceiling",),
    "uq": ("attack_steps", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "uq_joint": ("attack_steps", "percentiles", "purify_delta", "purify_steps", "batch_size"),
    "gibbs": ("attack_steps", "batch_size", "sweep_purify"),
    "sgld": ("attack_steps", "batch_size", "sweep_purify"),
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
    return out


def clean(cbm, datahandler, analysis, budgets: List[float], device) -> Dict[str, float]:
    jem = isinstance(cbm, JEMMLP)   # no exact log Z: no loss_gen on test
    m = evaluate(cbm, datahandler.classification[SPLIT], device,
                 log_Z=float("nan") if jem else None)
    names = ("acc", "loss_dis") if jem else ("acc", "loss_dis", "loss_gen")
    out = {key(name, SPLIT): m[name] for name in names}
    if analysis.rob_ceiling:
        X = datahandler.data[SPLIT].detach().cpu().numpy()
        y = datahandler.labels[SPLIT].detach().cpu().numpy()
        for eps in budgets:
            ceiling = robust_accuracy_ceiling(X, y, rel_to_abs(eps, range_size_of(cbm)))
            if ceiling == ceiling:  # nan: not two classes, or too many points
                out[key("rob_ceiling", SPLIT, eps)] = ceiling
    return out


def _uq_config(analysis, budgets, **kw) -> UQConfig:
    return UQConfig(
        norm="inf", eps_rel=list(budgets), attack_num_steps=analysis.attack_steps,
        percentiles=list(analysis.percentiles), delta_rel=list(analysis.purify_delta),
        num_steps=analysis.purify_steps, eval_batch_size=analysis.batch_size, **kw)


def _evaluate(cbm, datahandler, cfg: UQConfig, device, sweep_purifier=None) -> UQResults:
    r = UQEvaluation(cfg).evaluate(
        cbm, datahandler.classification[SPLIT], device,
        calib_loader=datahandler.classification["valid"], sweep_purifier=sweep_purifier)
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


def _attack_keys(r: UQResults, suffix: str = "") -> Dict[str, float]:
    out = {}
    for eps, acc in r.adv_accuracies.items():
        out[key(f"rob{suffix}", SPLIT, eps)] = acc
        out[key(f"log_px{suffix}", SPLIT, eps)] = float(r.adv_log_px[eps].mean())
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
    out = _attack_keys(r)
    out[key("log_px", SPLIT, 0)] = float(r.clean_log_px.mean())
    for pct, rate in r.clean_flagged.items():
        out[key("detect", SPLIT, 0, f"q{pct:g}")] = rate
    for delta, m in r.clean_purification_results.items():
        out[key("purify", SPLIT, 0, f"d{delta:g}")] = m.accuracy_after_purify
    return out


def uq_joint(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    cfg = _uq_config(analysis, budgets, attack_method="JOINT_PGD")
    return _attack_keys(_evaluate(cbm, datahandler, cfg, device), "_joint")


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


def gibbs(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    return _sweep("gibbs", cbm, datahandler, analysis, budgets, device)


def sgld(cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    p = analysis.sweep_purify
    purifier = SGLDPurification(step_rel=p.step, steps=p.sgld.steps, step_size=p.sgld.step_size,
                                noise_std=p.sgld.noise_std, batch_size=p.sgld.batch_size)
    return _sweep("sgld", cbm, datahandler, analysis, budgets, device, purifier)


def run_part(part: Callable, cbm, datahandler, analysis, budgets, device) -> Dict[str, float]:
    set_seed(SEED)
    return part(cbm, datahandler, analysis, budgets, device)
