import math
import os

from omegaconf import OmegaConf


def _training_regime(*, _root_):
    parts = []
    if OmegaConf.select(_root_, "trainer.nll") is not None:
        parts.append("nat")
    if OmegaConf.select(_root_, "trainer.adversarial") is not None:
        parts.append("at")
    return "_".join(parts) or "none"


_DTYPE_SUFFIX = {
    None: "", "float32": "", "float64": "",
    "complex64": "c64", "complex128": "c128",
}


def _dtype_suffix(*, _root_):
    dtype = OmegaConf.select(_root_, "born.init_kwargs.dtype")
    return _DTYPE_SUFFIX.get(dtype, f"_{dtype}")


def _outputs_root() -> str:
    root = os.environ.get("BM4TC_DATA_ROOT")
    return f"{root}/outputs" if root else "outputs"


def _alpha_suffix(*, _root_) -> str:
    """Kind-suffix token encoding the NLL alpha (e.g. 0.0->a0, 0.5->a05, 1.0->a1).

    Matches the manual naming convention (a0, a001, a01, a02, a05, a1) so that
    HPO sweeps over a CLI-supplied alpha land in distinct ``{experiment}`` output
    folders (and analysis mirrors) per alpha. Reads ``trainer.nll.alpha`` for NLL
    runs, falling back to ``trainer.adversarial.alpha`` for adversarial training.
    """
    alpha = OmegaConf.select(_root_, "trainer.nll.alpha")
    if alpha is None:
        alpha = OmegaConf.select(_root_, "trainer.adversarial.alpha")
    digits = str(float(alpha)).replace(".", "").rstrip("0")
    return "a" + (digits if digits else "0")


def _stage_path(*, _root_) -> str:
    """Path segment for the experiment stage (hpo/ | seed_sweep/).

    Returns ``"{stage}/"`` when a non-empty ``stage`` field is set, else ``""``.
    Keeps legacy (flat) configs without a stage at their original location.
    """
    stage = OmegaConf.select(_root_, "stage")
    return f"{stage}/" if stage else ""


def register_resolvers():
    if not OmegaConf.has_resolver("outputs_root"):
        OmegaConf.register_new_resolver("outputs_root", _outputs_root)
    if not OmegaConf.has_resolver("training_regime"):
        OmegaConf.register_new_resolver(
            "training_regime", _training_regime, use_cache=False
        )
    if not OmegaConf.has_resolver("complement_100"):
        OmegaConf.register_new_resolver("complement_100", lambda x: 100 - int(x))
    if not OmegaConf.has_resolver("complement_200"):
        OmegaConf.register_new_resolver("complement_200", lambda x: 200 - int(x))
    if not OmegaConf.has_resolver("dtype_suffix"):
        OmegaConf.register_new_resolver("dtype_suffix", _dtype_suffix, use_cache=False)
    if not OmegaConf.has_resolver("stage_path"):
        OmegaConf.register_new_resolver("stage_path", _stage_path, use_cache=False)
    if not OmegaConf.has_resolver("alpha_suffix"):
        OmegaConf.register_new_resolver("alpha_suffix", _alpha_suffix, use_cache=False)
    if not OmegaConf.has_resolver("geom_lr"):
        OmegaConf.register_new_resolver(
            "geom_lr",
            lambda alpha, lr_cls, lr_gen: math.exp(
                (1 - float(alpha)) * math.log(float(lr_cls))
                + float(alpha) * math.log(float(lr_gen))
            ),
        )
