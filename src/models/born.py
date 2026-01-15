import torch
from typing import *
from .classifier import BornClassifier
from .generator import BornGenerator
import src.utils.schemas as schemas
from omegaconf import OmegaConf
import src.utils.get as get
import logging
logger = logging.getLogger(__name__)


# Think about subclassing tk.models.TensorNetwork
class BornMachine:
    """
    Unified Born Machine that owns and synchronizes a classifier and generator.
    Handles shared tensors, device movement, checkpointing, and parameter access.
    """

    def __init__(self, cfg: schemas.BornMachineConfig,
                 data_dim: int | None = None, num_classes: int | None = None,
                 device: torch.device | None = None,
                 tensors: List[torch.Tensor] | None = None):
        """
        Initialize a BornMachine with shared classifier and generator.

        Args:
            cfg: Configuration for the Born machine (embedding, MPS init params).
            data_dim: Number of input features. Required if not in cfg.init_kwargs.
            num_classes: Number of output classes. Required if not in cfg.init_kwargs.
            device: Torch device for tensors.
            tensors: Pre-trained tensors to initialize from (for loading checkpoints).
        """
        # Edit init_kwargs if not fully instantiated yet.
        OmegaConf.set_struct(cfg, False) # Temporarily disable struct mode to set dynamic values
        if getattr(cfg.init_kwargs, "n_features", None) is None:
            if data_dim is None:
                raise ValueError(f"Give data dim explicitly or implicitly through configuration.")
            cfg.init_kwargs.n_features = data_dim + 1
        if getattr(cfg.init_kwargs, "out_dim", None) is None:
            if num_classes is None:
                raise ValueError(f"Give num classes explicitly or implicitly through configuration.")
            cfg.init_kwargs.out_dim = num_classes
        OmegaConf.set_struct(cfg, True) # Re-enable struct mode for safety
        
        # 1. Initialize embedding 
        self.embedding_name = cfg.embedding
        self.embedding = get.embedding(self.embedding_name)
        self.input_range = get.range_from_embedding(self.embedding_name)

        # 2. Intialize classifier, either from tensors, or configuration file
        if tensors is not None:
            self.classifier = BornClassifier(embedding=self.embedding, tensors=tensors)
        else:
            init_cfg = OmegaConf.to_object(cfg.init_kwargs)
            self.classifier = BornClassifier(embedding=self.embedding, device=device, **init_cfg)

        # Save out_position if not done yet.
        OmegaConf.set_struct(cfg, False)
        cfg.init_kwargs.out_position = self.classifier.out_position
        OmegaConf.set_struct(cfg, True)

        # 3. Create generator with shared tensors
        self.generator = BornGenerator(tensors=self.classifier.tensors, 
                                       embedding=self.embedding, 
                                       cls_pos=self.classifier.out_position,
                                       in_dim=self.classifier.in_dim[0],
                                       num_cls=cfg.init_kwargs.out_dim,
                                       input_range=self.input_range,
                                       device=device)

        # Save machine specific attributes, like embedding, physical dimension, input range,...
        (self.in_dim, self.out_dim,
         self.bond_dim, self.out_position) = (cfg.init_kwargs.in_dim, cfg.init_kwargs.out_dim,
                           cfg.init_kwargs.bond_dim, self.classifier.out_position)
        
        self.cfg = cfg
        self.device = device
        
    # ==========================================================
    # Forward APIs
    # ==========================================================
    def class_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities p(c|x) using the classifier.

        Args:
            data: Input tensor of shape (batch_size, data_dim).

        Returns:
            Tensor of shape (batch_size, num_classes) with normalized probabilities.
        """
        return self.classifier.probabilities(data)

    def sample(self, cfg: schemas.SamplingConfig, cls: Optional[int] = None) -> torch.Tensor:
        """
        Sample from the learned distribution x ~ p(x|c) using the generator.

        Args:
            cfg: Sampling configuration (num_bins, num_spc, batch_spc, method).
            cls: Class index to sample from. If None, samples from all classes.

        Returns:
            If cls is None: Tensor of shape (num_spc, num_classes, data_dim).
            If cls is int: Tensor of shape (num_spc, data_dim).
        """
        if cls is None:
            return self.generator.sample_all_classes(cfg)
        else:
            return self.generator.sample_single_class(cls, cfg)

    # ==========================================================
    # Parameter access / sync control
    # ==========================================================
    def parameters(self, mode: str = "classifier") -> Iterable[torch.nn.Parameter]:
        """
        Return parameters depending on training mode.
        mode ∈ {"classifier", "generator"}.
        """
        if mode == "classifier":
            return self.classifier.parameters()
        elif mode == "generator":
            return self.generator.parameters()
        else:
            raise ValueError(f"Unknown mode '{mode}' — choose 'classifier' or 'generator'.")

    def sync_tensors(self, after: str, verify: bool = False):
        """
        Synchronize tensors between classifier and generator views.

        After training one component, call this to copy its updated tensors
        to the other component. Critical for maintaining consistency.

        Args:
            after: Which training phase just completed.
                   "classification" -> copy classifier tensors to generator.
                   "generation" -> copy generator tensors to classifier.
            verify: If True, log verification that tensors match after sync.
        """
        # self.to("cpu")  # ensure host memory copy
        self.classifier.reset()
        self.generator.reset()

        src, dst = (self.classifier, self.generator) if after == "classification" else (self.generator, self.classifier)
        dst.initialize([t.clone() for t in src.tensors])

        if verify:
            for i, (t1, t2) in enumerate(zip(src.tensors, dst.tensors)):
                if not torch.allclose(t1, t2):
                    logger.error(f"[BornMachine] Tensor {i} mismatch (max diff: {(t1 - t2).abs().max().item():.3e})")
                else:
                    logger.debug(f"[BornMachine] Tensor {i} synchronized successfully.")

    # ==========================================================
    # Device management
    # ==========================================================
    def to(self, device: torch.device):
        """Move classifier, generator, and all tensors to the specified device."""
        self.classifier.to(device)
        self.generator.to(device)
        self.device = device
        return self

    # ==========================================================
    # Checkpoint handling
    # ==========================================================
    def save(self, path: str):
        """
        Save the BornMachine state to a file.

        Args:
            path: File path to save the checkpoint.
        """
        state = {
            "tensors": self.classifier.tensors,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "BornMachine":
        """
        Load a BornMachine from a checkpoint file.

        Args:
            path: File path to load the checkpoint from.

        Returns:
            A new BornMachine instance with loaded tensors and config.
        """
        checkpoint = torch.load(path)
        cfg: schemas.BornMachineConfig = OmegaConf.create(checkpoint["config"])
        born_machine = cls(cfg=cfg, tensors=checkpoint["tensors"])
        logger.info(f"[BornMachine] loaded from {path}")
        return born_machine

    def reset(self):
        """Reset internal state of both classifier and generator MPS networks."""
        self.classifier.reset()
        self.generator.reset()

    