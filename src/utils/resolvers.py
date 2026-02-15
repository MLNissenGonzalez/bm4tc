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


def register_resolvers():
    if not OmegaConf.has_resolver("training_regime"):
        OmegaConf.register_new_resolver(
            "training_regime", _training_regime, use_cache=False
        )
