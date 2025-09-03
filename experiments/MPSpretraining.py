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
from src.schemas import Config

from src.mps.categorisation import mps_cat_loader, disr_train_mps, mps_acc_eval
from src.datasets.preprocess import preprocess_pipeline
from src.datasets.gen_n_load import load_dataset, LabelledDataset
import tensorkrowch as tk
import torch
from omegaconf import OmegaConf


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Raw data loading
    dataset: LabelledDataset = load_dataset(cfg=cfg.dataset)
    dataset_name = dataset.name
    data_dim = dataset.num_feat
    num_cls = dataset.num_cls

    # 2. MPS intialization
    init_cfg = OmegaConf.to_object(cfg.model.mps.init_kwargs)  
    mps = tk.models.MPSLayer(n_features=data_dim+1,
                             out_dim=num_cls,
                             device=device,
                             **init_cfg
                             )
    # 3. Data preprocessing, 
    X, t, scaler = preprocess_pipeline(X_raw=dataset.X, t_raw=dataset.t, 
                                       split=cfg.dataset.split, 
                                       random_state=cfg.dataset.split_seed, 
                                       embedding=cfg.model.mps.embedding)
    
    # 4. Data embedding and data loaders
    loader = {}
    for split in ["train", "valid", "test"]:
        loader[split] = mps_cat_loader( X=X[split],
                                        t=t[split], 
                                        batch_size=cfg.pretrain.mps.batch_size,
                                        embedding=cfg.model.mps.embedding,
                                        phys_dim=cfg.model.mps.init_kwargs.in_dim,
                                        split=split)
    
    # 5. MPS pretraining
    best_tensors, train_loss, val_accuracy = disr_train_mps(mps=mps, loaders=loader,
                                                            cfg=cfg.pretrain.mps,
                                                            device=device,
                                                            title=dataset_name)
    mps = tk.models.MPS(tensors=best_tensors)

    test_accuracy = mps_acc_eval(mps, loader["test"], device)

if __name__ == "__main__":
    main()
