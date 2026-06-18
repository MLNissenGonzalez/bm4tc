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
    if not OmegaConf.has_resolver("geom_lr"):
        OmegaConf.register_new_resolver(
            "geom_lr",
            lambda alpha, lr_cls, lr_gen: math.exp(
                (1 - float(alpha)) * math.log(float(lr_cls))
                + float(alpha) * math.log(float(lr_gen))
            ),
        )
