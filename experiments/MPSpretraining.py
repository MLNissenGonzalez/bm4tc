"""
Pretraining for MPS only script. Mainly for debugging as pretrained MPS is not of interest per se.
1. Raw dataset generated or loaded.
2. MPS initialized
3. Dateset preprocessed (depends on embedding used)
4. MPS trained as classifier
Log pretraining such that visualisation is possible after the training.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))  # make src importable


import hydra
from omegaconf import DictConfig, OmegaConf
from src.schemas import Config  
from src.mps.utils import mps_cat_loader, disr_train_mps
from src.datasets.two_dim_toydata import raw_data_gen, preprocess_pipeline
import tensorkrowch as tk
import torch

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Raw data gen
    X, t = raw_data_gen(title=cfg.dataset.title, 
                        size=cfg.dataset.size, 
                        noise=cfg.dataset.noise, 
                        random_state=cfg.dataset.seed,
                        factor=cfg.dataset.factor)
    
    # 2. MPS intialization
    mps = tk.models.MPSLayer(n_features=(cfg.dataset.n_feat+1),
                             in_dim=cfg.model_mps.in_dim,
                             out_dim=cfg.dataset.n_cls,
                             bond_dim=cfg.model_mps.bond_dim,
                             out_position=cfg.model_mps.out_position,
                             boundary=cfg.model_mps.boundary,
                             init_method=cfg.model_mps.init_method,
                             device=device
                             )
    
    # 3. Data preprocessing, 
    X, t, scaler = preprocess_pipeline(X=X, t=t, 
                                       split=cfg.dataset.split, 
                                       random_state=cfg.dataset.seed, 
                                       embedding=cfg.model_mps.embedding)
    
    # 4. Data embedding and data loaders
    loader = {}
    for split in ["train", "valid", "test"]:
        loader[split] = mps_cat_loader( X=X[split],
                                        t=t[split], 
                                        batch_size=cfg.pretrain_mps.batch_size,
                                        embedding=cfg.model_mps.embedding,
                                        phys_dim=cfg.model_mps.in_dim,
                                        split=split)
    
    # TODO: Include configuration schemsa in training functions
    # 5. MPS pretraining
    best_tensors, train_loss, val_accuracy = disr_train_mps(mps=mps, loaders=loader,
                                                            cfg=cfg.pretrain_mps,
                                                            cls_pos=cfg.model_mps.out_position,
                                                            device=device, 
                                                            phys_dim=cfg.model_mps.in_dim,
                                                            title=cfg.dataset.title)
    

if __name__ == "__main__":
    main()
