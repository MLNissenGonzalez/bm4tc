from omegaconf import OmegaConf


def _training_regime(*, _root_):
    parts = []
    for key, code in [
        ("trainer.classification", "cls"),
        ("trainer.generative", "gen"),
        ("trainer.adversarial", "adv"),
        ("trainer.ganstyle", "gan"),
    ]:
        if OmegaConf.select(_root_, key) is not None:
            parts.append(code)
    return "".join(parts) or "none"


_DTYPE_SUFFIX = {
    None: "", "float32": "", "float64": "",
    "complex64": "c64", "complex128": "c128",
}


def _dtype_suffix(*, _root_):
    dtype = OmegaConf.select(_root_, "born.init_kwargs.dtype")
    return _DTYPE_SUFFIX.get(dtype, f"_{dtype}")


def register_resolvers():
    if not OmegaConf.has_resolver("training_regime"):
        OmegaConf.register_new_resolver(
            "training_regime", _training_regime, use_cache=False
        )
    if not OmegaConf.has_resolver("complement_100"):
        OmegaConf.register_new_resolver("complement_100", lambda x: 100 - int(x))
    if not OmegaConf.has_resolver("dtype_suffix"):
        OmegaConf.register_new_resolver("dtype_suffix", _dtype_suffix, use_cache=False)
