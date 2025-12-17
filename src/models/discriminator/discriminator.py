# TODO: Handle Imports.
import torch
from torch import autograd, nn
from typing import *
import src.utils.schemas as schemas
from torch.utils.data import DataLoader
from src.data.handler import DataHandler
import src.utils.get as get
from src.models.born import BornMachine
from backbones import BackBone
import wandb
from . import get_backbone, get_head
import logging
from copy import deepcopy
logger = logging.getLogger(__name__)


# TODO: Include training in this? Yes because, discriminator should be understood to be only an approximation to distance on probability spaces
# TODO: criterion, some basic data info, architecture design, are owned by the class. training info has to fed to the discriminator

# ADDED AS ISSUE.
# TODO: Add other discriminator functions, e.g. MPS with MLP module, shared backbone critic with class heads
# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS

class Critic(nn.Module):
    def __init__(self, cfg: schemas.GANStyleConfig, datahandler: DataHandler, 
                 backbone: BackBone | None = None, device: torch.device | None = None):
        super().__init__()
        self.num_cls, self.data_dim = datahandler.num_cls, datahandler.data_dim
        self.model_cfg = cfg.critic
        self.sampling_cfg = cfg.sampling

        # backbone maps (N, D) -> (N, F), could also be the bornmachine
        if backbone is None:
            self.backbone = get_backbone(cfg.critic.backbone.architecture, self.data_dim,
                                         False, **cfg.critic.backbone.model_kwargs)
        else:
            self.backbone = backbone
        
        self.bottleneck_dim = self.backbone.out_dim
        # heads as separate params, either class aware (GAN-style training only) or class agnostic (necessary for Adv.Train)
        self.checkpoint()
        self.class_aware = cfg.critic.head.class_aware
        self.head = get_head(cfg.critic.head.class_aware, cfg.critic.head.architecture, cfg.critic.head.model_kwargs,
                             self.bottleneck_dim, self.num_cls)
        
        if self.class_aware:
            self._forward_impl: Callable[[torch.FloatTensor], torch.FloatTensor] = self.aware_forward
        else:
            self._forward_impl : Callable[[torch.FloatTensor], torch.FloatTensor] = self.agnostic_forward

        # Distances / losses
        self.criterion_name = cfg.critic.criterion.name.lower().replace(" ",
                                                                        "").replace("-", "")
        wasserstein_names = ["wasserstein", "wgangp", "wgan"]
        self.lamb = None
        if self.criterion_name in wasserstein_names:
            self.lamb: float = self.model_cfg.criterion.kwargs.get(
                "lamb", 10.0)
            self._loss = self.wgan
            self._generator_loss = self.gen_wgan  
        else:
            self._loss = self.bce
            self.swapped = self.model_cfg.criterion.kwargs.get("swapped", True)
            if self.swapped:
                self._generator_loss = self.gen_swapped_bce
            else:
                self._generator_loss = self.gen_bce


        # Initialize optimizer here, as critic train will be called multiple times.
        self.train_cfg = cfg.critic.discrimination
        self.optimizer = get.optimizer(
            self.parameters(), self.train_cfg.optimizer)

        self.device = device if device is not None else "cpu"
        self.to(device=self.device)

    def checkpoint(self):
        self.backbone.reset()
        self.best_state = self.state_dict()

    def aware_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  
        B, C, D = x.shape # x: (B, C, D)
        # class samples are independently disciminated as natural or generated
        feats = self.backbone.forward(x.reshape(B*C, D)).reshape(B, C, -1)
        logits = self.head.forward(feats)       # logits: (B, C)
        return logits.reshape(B*C)              # returns: (B*C,)
    
    def agnostic_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if len(x.shape) == 3:
            N, C, D = x.shape
            x = x.reshape(N*C, D)
        feats = self.backbone.forward(x)    # x: (N', D), feats: (N', F)
        logits = self.head.forward(feats)   # logits: (N',)
        return logits
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._forward_impl(x) # returns (N,)
    

    # WGAN for ganstyle training only, increasing the generative capabilities. Thus, assume input to come by class.
    def wgan(self,
             natural: torch.FloatTensor,        # (N, C, D)
             generated: torch.FloatTensor       # (N, C, D)
             ):  
        loss = (self.forward(generated).mean()  # aggregation over all classes works also in the ensemble case here
                - self.forward(natural).mean()
                + self.lamb * self._gradient_penality(natural, generated))
        return loss
    
    # for WGAN-GP critic
    def _gradient_penality(self,
                           natural: torch.FloatTensor,      # (N, C, D)
                           generated: torch.FloatTensor     # (N, C, D)
                           ):  
        N, C, D = natural.shape
        alpha = torch.rand(size = (N, C, 1), device=natural.device)                 # (N, C, 1)
        x_hat = alpha * natural + (1-alpha) * generated                             # (N, C, D)
        x_hat.requires_grad = True
        output = self.forward(x_hat).mean()                                         # scalar
        grads = autograd.grad(outputs=output, inputs=x_hat, create_graph=True)[0]   # (N, C, D)
        penalty = torch.square(
            1.0-grads.norm(p=2.0, dim=-1)).mean()                                         # grads.norm (N,C)
        return penalty                                                     # scalar

    def gen_wgan(self, generated: torch.FloatTensor):
        loss = -self.forward(generated).mean() # notice the relative minus sign
        return loss
    
    # Can be used for ganstyle training or adversarial training
    def bce(self,
            natural: torch.FloatTensor,     # (N, C, D) or (N, C)
            generated: torch.FloatTensor,    # (N, C, D),
            eps: float = 1e-12
            ):
        
        nat = -torch.log(torch.sigmoid(self.forward(natural)).clamp(min=eps))      # (N*C, ) or (N',)
        gen = -torch.log((1-torch.sigmoid(self.forward(generated))).clamp(min=eps))  # (N*C, ) or (N',)
        loss = nat.mean() + gen.mean()      # scalar
        return loss

    def gen_swapped_bce(self, generated: torch.FloatTensor, eps: float = 1e-12):
        loss = -torch.log(torch.sigmoid(self.forward(generated)).clamp(min=eps))
        return loss
    
    def gen_bce(self, generated: torch.FloatTensor, eps: float = 1e-12):
        loss = torch.log((1-torch.sigmoid(self.forward(generated))).clamp(min=eps))
        return loss

    def loss(self, natural: torch.FloatTensor, generated: torch.FloatTensor):
        return self._loss(natural, generated)
        
    def generator_loss(self, generated: torch.FloatTensor):
        return self._generator_loss(generated)

    def train_step(self, naturals: torch.FloatTensor, generated: torch.FloatTensor):
        self.optimizer.zero_grad()
        loss = self.loss(naturals, generated)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()
    
    def _check(self, patience_counter: int, best_loss: float, epoch_loss: float):
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            self.checkpoint()
        else:
            patience_counter += 1

        if patience_counter >= self.train_cfg.patience:
            stop = True
        else: stop = False

        return stop, best_loss, patience_counter  
    
    def discriminate(self, datahandler: DataHandler, loaders: Dict[str, DataLoader], mode: str) -> float:
        
        modes = ["train", "valid", "test"]
        if mode not in modes: raise KeyError(f"'mode' can be 'train', 'valid', or 'test'.")
        self.train() if mode == "train" else self.eval()

        losses = []
        for naturals, generated in zip(
            datahandler.discrimination[mode], loaders[mode]):    # (num_spc, C, D)
            naturals : torch.FloatTensor = naturals.to(self.device)
            generated: torch.FloatTensor = generated.to(self.device)
            if mode=="train":
                loss = self.train_step(naturals, generated)
            else:
                with torch.no_grad():
                    loss = self.loss(naturals, generated).cpu().item()
            losses.append(loss)
        epoch_loss = sum(losses) / len(losses)
        return epoch_loss

    # Generator held fixed, only inner optimization of critic performed.
    def ganstyle_pretrain(self, datahandler: DataHandler, bornmachine: BornMachine, 
                          device: torch.device, loaders_path: str | None = None):
        logger.info("Pretraining critic.")
        # Checking what is trainable
        if self.backbone.pretrained:
            self.backbone.freeze()
        self.to(device=device)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = get.optimizer(trainable_params, self.train_cfg.optimizer)

        # Configuring experiment 
        max_epoch = self.train_cfg.max_epoch_pre
        wandb.define_metric("gan/dis_train/loss", summary=None)
        wandb.define_metric("gan/dis_valid/loss", summary=None)
        patience_counter, best_loss = 0, float("inf")
        
        # Creating or loading a dataset to pretrain the discriminator
        if loaders_path is None:
            synth_loaders = self.create_loaders_of_synthetics(datahandler, bornmachine, device)
        else: synth_loaders = torch.load(loaders_path)

        # Epoch loop
        for epoch in range(max_epoch):
            train_loss = self.discriminate(datahandler, synth_loaders, "train")
            valid_loss = self.discriminate(datahandler, synth_loaders, "valid")
            stop, best_loss, patience_counter = self._check(patience_counter, best_loss, valid_loss)
            wandb.log(
                {
                    "gan/dis_train/loss": train_loss,
                    "gan/dis_valid/loss": valid_loss
                }
            )
            if stop:
                logger.info(f"Stopped training early at epoch {epoch + 1}.")
                break

        # Completing pretraining    
        self.pretrained = True
        self.backbone.unfreeze()
        self.backbone.reset()
        self.load_state_dict(self.best_state)
        self.optimizer = get.optimizer(self.parameters(), self.train_cfg.optimizer)
        logger.info("Pretraining of critic completed.")

    # for pretraining only
    def create_loaders_of_synthetics(self, datahandler: DataHandler, 
                                      bornmachine: BornMachine, device: torch.device) -> Dict[str, DataLoader]:
        loaders = {}
        bornmachine.to(device)
        sampling_cfg = deepcopy(self.sampling_cfg)
        for i, split in enumerate(["train", "valid", "test"]):
            sampling_cfg.num_spc = datahandler.num_spc[i]
            with torch.no_grad():
                generated = bornmachine.sample(sampling_cfg).cpu()
            loaders[split] = DataLoader(generated, batch_size=self.train_cfg.batch_size,
                                        shuffle=(split=="train"), drop_last=(split=="train"))
        return loaders

    def adversarial_pretrain(self, datahandler: DataHandler):
        return NotImplementedError