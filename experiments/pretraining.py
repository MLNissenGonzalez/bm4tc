"""
Pretraining script. Could be used to seperate the experiment into two: Pretraining and adversarial training. Also good for debugging, I guess.

1. Dataset loaded (labelled dataset, e.g. 2 moons)
2. MPS initialized
3. Dateset preprocessed (depends on embedding used)
4. MPS trained as classifier
5. Discriminator initialized
6. MPS generates data for discriminator
7. Discrimination pretraining dataset preloaded (real and synthesised samples)
8. Discriminator pretrained (binary classification problem)

Log pretraining such that visualisation is possible after the training.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))  # make src importable

import hydra
from src.schemas import Config

from src.mps.categorisation import mps_cat_loader, cat_train_mps, mps_acc_eval
from src.mps.utils import batch_sampling_mps
from src.datasets.preprocess import preprocess_pipeline
from src.datasets.gen_n_load import load_dataset, LabelledDataset
from src.discriminator.utils import MLPdis, dis_pretrain_loader, discriminator_pretraining
import tensorkrowch as tk
import torch
from omegaconf import OmegaConf
import logging
import numpy as np

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):

    device = torch.device("cpu") # change later when device calls are tracked

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
    cls_pos = mps.out_features[0]

    # 3. Data preprocessing, 
    X, t, scaler = preprocess_pipeline(X_raw=dataset.X, t_raw=dataset.t, 
                                       split=cfg.dataset.split, 
                                       random_state=cfg.dataset.split_seed, 
                                       embedding=cfg.model.mps.embedding)
    
    # 4. Data embedding and data loaders
    loader = {}
    size_per_class = {}
    for split in ["train", "valid", "test"]:
        loader[split] = mps_cat_loader( X=X[split],
                                        t=t[split], 
                                        batch_size=cfg.pretrain.mps.batch_size,
                                        embedding=cfg.model.mps.embedding,
                                        phys_dim=cfg.model.mps.init_kwargs.in_dim,
                                        split=split)
        size_per_class[split] = t[split].shape[0] // num_cls
        logging.debug(f"{size_per_class[split]=}")
    
    # 5. MPS pretraining
    best_tensors, train_loss, val_accuracy = cat_train_mps(mps=mps, loaders=loader,
                                                            cfg=cfg.pretrain.mps,
                                                            device=device,
                                                            title=dataset_name)
    mps = tk.models.MPS(tensors=best_tensors)

    test_accuracy = mps_acc_eval(mps, loader["test"], device)
    
    logger.info("MPS pretraining done.")

    # 6. Discriminator initialization
    dis = MLPdis(cfg.model.dis, input_dim=data_dim)

    # 7. Synthezesing and wrapping to data loader
    X_synth = {}
    splits = ["train", "valid", "test"]
    dis_loaders = {}
    for split in range(3):
        n_spc = np.floor(np.array(cfg.dataset.split) * cfg.dataset.gen_dow_kwargs.size)[split]
        logger.debug(f"{n_spc=}")
        n_spc = n_spc.astype(int)
        X_synth[splits[split]] = batch_sampling_mps(
            mps=mps, embedding=cfg.model.mps.embedding,
            cls_pos=cls_pos,
            num_spc=n_spc,
            num_bins=cfg.gantrain.num_bins,
            batch_spc=cfg.gantrain.n_synth,
            device=device
        ).detach() # We do not want MPS gradients.

        split = splits[split]
        dis_loaders[split] = dis_pretrain_loader(X_real=X[split], 
                                          c_real=t[split],
                                          X_synth = X_synth[split],
                                          mode=cfg.model.dis.mode,
                                          batch_size=cfg.pretrain.dis.batch_size,
                                          split=split
                                          )
    logger.info("Data for pretraining of discriminator loaded.")

    # 8. Discriminator pretraining
    dis_pretrain_results = {}
    for i in dis_loaders["train"].keys():
        dis_pretrain_results[i] = discriminator_pretraining(dis=dis, 
                                                            cfg=cfg.pretrain.dis,
                                                            loader_train=dis_loaders["train"][i],
                                                            loader_test=dis_loaders["valid"][i])
    logger.info("Pretraining completed.")
    
if __name__ == "__main__":
    main()
