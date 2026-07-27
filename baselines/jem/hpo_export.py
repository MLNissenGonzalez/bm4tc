"""Export Optuna's best JEM trial into the matching seed-sweep config."""

from __future__ import annotations

import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

try:
    from hydra.experimental.callback import Callback
except ModuleNotFoundError:  # Allows testing the pure YAML helpers without Hydra.
    class Callback:  # type: ignore[no-redef]
        pass


_ALPHA_TARGETS = {
    0.0: "a0",
    0.01: "a001",
    0.1: "a01",
    0.2: "a02",
    0.5: "a05",
    1.0: "a1",
}


def target_for_alpha(alpha: float) -> str:
    """Map the configured alpha ladder to its seed-sweep YAML stem."""
    for candidate, target in _ALPHA_TARGETS.items():
        if math.isclose(alpha, candidate, rel_tol=0.0, abs_tol=1e-12):
            return target
    supported = ", ".join(str(value) for value in _ALPHA_TARGETS)
    raise ValueError(f"No seed-sweep config for alpha={alpha}; expected one of {supported}.")


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    return yaml.safe_dump(value, default_flow_style=True).strip()


def patch_seed_sweep_config(path: Path, best_params: dict[str, Any]) -> list[str]:
    """Replace only optimized dotted keys, preserving comments and layout."""
    text = path.read_text()
    updated = []
    for key, value in best_params.items():
        pattern = re.compile(
            rf"^(\s*{re.escape(key)}:\s*)([^#\n]*?)(\s+#.*)?$",
            flags=re.MULTILINE,
        )

        def replacement(match: re.Match[str]) -> str:
            suffix = match.group(3) or ""
            return f"{match.group(1)}{_yaml_scalar(value)}{suffix}"

        text, count = pattern.subn(replacement, text)
        if count != 1:
            raise KeyError(
                f"Expected exactly one {key!r} entry in {path}, found {count}."
            )
        updated.append(key)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
    return updated


class HPOExportCallback(Callback):
    """Hydra callback that propagates a completed Optuna sweep automatically."""

    def __init__(self, target: str):
        self.target = target
        self._sweep_dir: Path | None = None

    @staticmethod
    def _current_sweep_dir() -> Path:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import to_absolute_path

        return Path(to_absolute_path(str(HydraConfig.get().sweep.dir)))

    def on_multirun_start(self, config, **kwargs: Any) -> None:
        self._sweep_dir = self._current_sweep_dir()

    def on_multirun_end(self, config, **kwargs: Any) -> None:
        sweep_dir = self._sweep_dir or self._current_sweep_dir()
        result_path = sweep_dir / "optimization_results.yaml"
        if not result_path.exists():
            raise FileNotFoundError(
                f"Optuna result not found at {result_path}; seed config was not changed."
            )
        result = yaml.safe_load(result_path.read_text()) or {}
        best_params = result.get("best_params")
        if not isinstance(best_params, dict) or not best_params:
            raise ValueError(f"No single-objective best_params found in {result_path}.")

        target = self.target
        if target == "auto_alpha":
            target = target_for_alpha(float(config.trainer.alpha))
        config_path = (
            Path(__file__).resolve().parent
            / "configs"
            / "experiment"
            / "seed_sweep"
            / f"{target}.yaml"
        )
        updated = patch_seed_sweep_config(config_path, best_params)
        print(
            f"[JEM HPO] Exported {len(updated)} best parameters to {config_path}: "
            + ", ".join(updated)
        )
