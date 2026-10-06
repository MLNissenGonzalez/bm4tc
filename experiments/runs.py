"""Studies, jobs and runs: the one owner of run names, run directories and run.json.

A study (``configs/studies/<name>.yaml``, merged onto ``configs/defaults.yaml``)
is a grid of cells (embedding x arch x alpha [x eps for AT]) times seeds. Each
(cell, seed) is a :class:`Job`; :meth:`Job.compose` builds its run config with the
Hydra compose API on the schema (D25).

Names follow D14: identity axes only, as prefix + value (``d3r40``, ``a0.01``,
``eps0.1``, ``s3``). A run lives at::

    {data root}/outputs/{study}/{embedding}/{arch}/a{alpha}[/eps{eps}]/s{seed}/

and is finished when it holds ``run.json`` (D17): identity, warm-start source,
git sha, launch time, the resolved config and its hash. Nothing parses run paths;
runs are found by reading manifests.
"""

import dataclasses
import datetime
import hashlib
import itertools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict

from experiments.config import register
from src.utils.paths import data_root

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
REGIMES = ("nat", "at")

register()


def outputs_root() -> Path:
    return data_root() / "outputs"


# ── Study schema ────────────────────────────────────────────────────────────

@dataclass
class GridConfig:
    embedding: List[str] = MISSING
    arch: List[str] = MISSING          # d{in_dim}r{bond_dim}, e.g. d3r40
    alpha: List[float] = MISSING
    eps: List[float] = MISSING         # AT training radius (eps_rel); NAT ignores it


@dataclass
class HPOConfig:
    n_trials: int = MISSING
    space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GibbsConfig:
    enabled: bool = MISSING
    sweeps: List[int] = MISSING
    num_bins: int = MISSING
    step: float = MISSING
    batch_size: int = MISSING
    subsample: Optional[int] = MISSING


@dataclass
class AnalysisConfig:
    """What `analyse` computes per run; values in configs/defaults.yaml (D20)."""
    attack_steps: int = MISSING
    joint_attack: bool = MISSING
    rob_ceiling: bool = MISSING
    percentiles: List[float] = MISSING
    purify_delta: List[float] = MISSING
    purify_steps: int = MISSING
    batch_size: int = MISSING
    gibbs: GibbsConfig = field(default_factory=GibbsConfig)


@dataclass
class StudyConfig:
    dataset: str = MISSING             # an option of configs/dataset/
    regime: str = MISSING              # nat | at
    init: str = "cold"                 # cold | warm; AT is always warm (D19)
    warm_from: Optional[str] = None    # the study holding the alpha=0 NAT runs; default {dataset}_nat
    seeds: Any = MISSING               # n (seeds 1..n) or a list of seeds
    grid: GridConfig = field(default_factory=GridConfig)
    embeddings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)   # fixed run-config values
    hpo: Optional[HPOConfig] = None    # None: no HPO, the study fixes every hparam
    budgets: List[float] = MISSING
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


# ── Names ───────────────────────────────────────────────────────────────────

_ARCH = re.compile(r"d(\d+)r(\d+)")


def parse_arch(arch: str) -> tuple[int, int]:
    """``d3r40`` -> (in_dim 3, bond_dim 40)."""
    m = _ARCH.fullmatch(arch)
    if m is None:
        raise ValueError(f"arch {arch!r} is not d<in_dim>r<bond_dim>")
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True)
class Cell:
    embedding: str
    arch: str
    alpha: float
    eps: Optional[float] = None        # AT only

    @property
    def name(self) -> str:
        """``legendre/d3r40/a0.01[/eps0.1]``: the cell's path below its study."""
        name = f"{self.embedding}/{self.arch}/a{self.alpha:g}"
        return name if self.eps is None else f"{name}/eps{self.eps:g}"

    def values(self) -> Dict[str, Any]:
        """The run-config values this cell fixes."""
        in_dim, bond_dim = parse_arch(self.arch)
        values = {
            "born.embedding": self.embedding,
            "born.init_kwargs.in_dim": in_dim,
            "born.init_kwargs.bond_dim": bond_dim,
            "trainer.alpha": self.alpha,
        }
        if self.eps is not None:
            values["trainer.evasion.eps_rel"] = [self.eps]
        return values


# ── Studies and jobs ────────────────────────────────────────────────────────

class Study:
    """A study file merged onto ``configs/defaults.yaml``; unknown keys fail."""

    def __init__(self, name: str):
        self.name = name
        path = CONFIGS / "studies" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No study {name!r} at {path}")
        self.cfg: StudyConfig = OmegaConf.merge(
            OmegaConf.structured(StudyConfig),
            OmegaConf.load(CONFIGS / "defaults.yaml"),
            OmegaConf.load(path),
        )
        if self.cfg.regime not in REGIMES:
            raise ValueError(f"{name}: regime must be one of {REGIMES}, got {self.cfg.regime!r}")
        if self.cfg.init not in ("cold", "warm"):
            raise ValueError(f"{name}: init must be cold or warm, got {self.cfg.init!r}")
        if self.cfg.regime == "at" and self.cfg.init != "warm":
            raise ValueError(f"{name}: AT runs are always warm (D19); set init: warm")

    @property
    def regime(self) -> str:
        return self.cfg.regime

    @property
    def warm(self) -> bool:
        return self.cfg.init == "warm"

    @property
    def warm_from(self) -> str:
        return self.cfg.warm_from or f"{self.cfg.dataset}_nat"

    def seeds(self) -> List[int]:
        seeds = self.cfg.seeds
        return list(range(1, seeds + 1)) if isinstance(seeds, int) else list(seeds)

    def cells(self) -> List[Cell]:
        g = self.cfg.grid
        eps = list(g.eps) if self.regime == "at" else [None]
        return [Cell(e, a, float(al), None if ep is None else float(ep))
                for e, a, al, ep in itertools.product(g.embedding, g.arch, g.alpha, eps)]

    def jobs(self) -> List["Job"]:
        return [Job(self, cell, seed) for cell in self.cells() for seed in self.seeds()]

    @cached_property
    def _hparams(self) -> Dict[str, Dict[str, Any]]:
        path = CONFIGS / "hparams" / f"{self.name}.yaml"
        if not path.exists():
            return {}
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    def hparams(self, cell: Cell) -> Dict[str, Any]:
        """The selected hparams of a cell (D21, D34); every key of the HPO space."""
        if self.cfg.hpo is None:
            return {}
        wanted = set(self.cfg.hpo.space)
        got = self._hparams.get(cell.name, {})
        missing = wanted - set(got)
        if missing:
            raise LookupError(
                f"{self.name}: no selected hparams for cell {cell.name} "
                f"(missing {sorted(missing)} in configs/hparams/{self.name}.yaml); "
                f"run HPO and `select {self.name}` first"
            )
        return {k: got[k] for k in sorted(wanted)}


class RunConflict(RuntimeError):
    """A finished run with a different config sits where a job would write."""


@dataclass(frozen=True)
class Job:
    study: Study
    cell: Cell
    seed: int

    @property
    def run_dir(self) -> Path:
        return outputs_root() / self.study.name / self.cell.name / f"s{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.study.name}/{self.cell.name}/s{self.seed}"

    def identity(self) -> Dict[str, Any]:
        c = self.cell
        return {"study": self.study.name, "dataset": self.study.cfg.dataset,
                "regime": self.study.regime, "embedding": c.embedding, "arch": c.arch,
                "alpha": c.alpha, "eps": c.eps, "seed": self.seed}

    def wandb(self) -> Dict[str, str]:
        """W&B grouping (D47): one group per grid cell, its seeds as runs."""
        return {"group": f"{self.study.name}/{self.cell.name}", "name": f"s{self.seed}",
                "job_type": "train"}

    def values(self, hparams: bool = True) -> Dict[str, Any]:
        """Every run-config value this job sets, in order of precedence (later wins)."""
        values = {
            **self.cell.values(),
            **self.study.cfg.embeddings.get(self.cell.embedding, {}),
            **self.study.cfg.config,
            **(self.study.hparams(self.cell) if hparams else {}),
            "tracking.seed": self.seed,
        }
        return OmegaConf.to_container(OmegaConf.create(values), resolve=True)

    def compose(self, hparams: bool = True) -> DictConfig:
        """The run config: config.yaml with the study's dataset and regime, then
        this job's values set key by key on the schema (an unknown key fails)."""
        with initialize_config_dir(str(CONFIGS), version_base=None):
            cfg = compose("config", overrides=[
                f"dataset={self.study.cfg.dataset}", f"trainer={self.study.regime}",
            ])
        for key, value in self.values(hparams).items():
            # The parent must exist and not be None: setting a key below NAT's
            # `evasion: null` would otherwise create an attack. Below a typed node
            # the schema (struct mode) rejects unknown keys; open dicts such as
            # optimizer.kwargs take new ones.
            parent = key.rpartition(".")[0]
            node = OmegaConf.select(cfg, parent, default=None) if parent else cfg
            if not isinstance(node, DictConfig):
                raise KeyError(f"{self.study.name}: {key!r}: no config node {parent!r} "
                               f"in a {self.study.regime} run")
            if dataclasses.is_dataclass(OmegaConf.get_type(node)):
                OmegaConf.update(cfg, key, value, merge=False)
            else:
                with open_dict(node):
                    OmegaConf.update(cfg, key, value, merge=False)
        return cfg

    # ── Warm start (D19) ────────────────────────────────────────────────────

    def warm_source(self) -> Optional[Dict[str, str]]:
        """The finished alpha=0 NAT run of the same dataset/embedding/arch/seed in
        the study's ``warm_from`` study, as ``{"run": dir, "hash": config hash}``;
        None for a cold job."""
        if not self.study.warm:
            return None
        want = {"study": self.study.warm_from, "dataset": self.study.cfg.dataset,
                "regime": "nat", "embedding": self.cell.embedding,
                "arch": self.cell.arch, "alpha": 0.0, "seed": self.seed}
        found = find_runs(**want)
        if len(found) != 1:
            raise LookupError(
                f"{self.name}: warm start needs exactly one finished run matching "
                f"{want}, found {len(found)}" + (f": {[str(d) for d, _ in found]}" if found else "")
                + f"; train {self.study.warm_from} first"
            )
        run_dir, manifest = found[0]
        return {"run": str(run_dir), "hash": manifest["config_hash"]}

    # ── run.json (D17) ──────────────────────────────────────────────────────

    def config_hash(self, cfg: DictConfig, init: Optional[Dict[str, str]]) -> str:
        """What makes two runs the same run: the resolved config (W&B settings
        aside) and the warm start it continues from."""
        content = OmegaConf.to_container(cfg, resolve=True)
        content.pop("tracking")
        content["seed"] = self.seed
        content["init"] = init and init["hash"]
        blob = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def claim(self, config_hash: str, replace: bool = False) -> bool:
        """Make the run dir ready for this job; False if it is already done.

        - finished with the same config hash: done, skip (a relaunch resumes);
        - finished with another hash: raise :class:`RunConflict`, unless
          ``replace``, which first moves the old run to
          ``outputs/{study}/.replaced/{date}/...`` (deleted only by ``prune``);
        - started but unfinished (no run.json): cleared and restarted.
        """
        run_dir = self.run_dir
        manifest = run_dir / "run.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())["config_hash"]
            if old == config_hash:
                return False
            if not replace:
                raise RunConflict(
                    f"{run_dir} holds a finished run with config {old}; this job's is "
                    f"{config_hash} (the study, its hparams, the defaults or the warm "
                    f"start changed). Pass --replace to move the old run to "
                    f"{outputs_root() / self.study.name / '.replaced'}/."
                )
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = outputs_root() / self.study.name / ".replaced" / stamp / self.cell.name / f"s{self.seed}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), str(dest))
        elif run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        return True

    def write_manifest(self, cfg: DictConfig, init: Optional[Dict[str, str]],
                       config_hash: str, launched: str, result: Dict[str, Any]) -> None:
        """Mark the run finished: written last, after the checkpoint."""
        manifest = {
            "identity": self.identity(),
            "cell": self.cell.name,
            "init": init,
            "git": _git_version(),
            "launched": launched,
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            "config_hash": config_hash,
            "result": result,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        (self.run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str))


def find_runs(**identity) -> List[tuple[Path, Dict[str, Any]]]:
    """Finished runs whose run.json identity matches every given axis (D17).
    ``study`` narrows the search to that study's directory."""
    root = outputs_root()
    if "study" in identity:
        root = root / identity["study"]
    found = []
    for manifest in sorted(root.rglob("run.json")) if root.exists() else []:
        if ".replaced" in manifest.parts:
            continue
        data = json.loads(manifest.read_text())
        if all(data["identity"].get(k) == v for k, v in identity.items()):
            found.append((manifest.parent, data))
    return found


def _git_version() -> str:
    """HEAD's sha, with ``+dirty`` when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout
        return sha + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
