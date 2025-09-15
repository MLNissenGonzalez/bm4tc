"""
Pretraining for MPS only script. Mainly for debugging as pretrained MPS is not of interest per se.
1. Raw dataset generated or loaded.
2. MPS initialized
3. Dateset preprocessed (depends on embedding used)
4. MPS trained as classifier
Log pretraining such that visualisation is possible after the training.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

import logging
from omegaconf import OmegaConf
import torch
import tensorkrowch as tk
import hydra

from src.schemas import Config
from src.datasets.gen_n_load import load_dataset, LabelledDataset
from src.datasets.preprocess import preprocess_pipeline
import src.mps.categorisation as mps_cat
from src._utils import _class_wise_dataset_size


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):

    device = torch.device("cpu")  # change later when device calls are tracked

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
        loader[split] = mps_cat.loader_creator(X=X[split],
                                               t=t[split],
                                               batch_size=cfg.pretrain.mps.batch_size,
                                               embedding=cfg.model.mps.embedding,
                                               phys_dim=cfg.model.mps.init_kwargs.in_dim,
                                               split=split)
        size_per_class[split] = _class_wise_dataset_size(t[split])
        logging.debug(f"{size_per_class[split]=}")

    # 5. MPS pretraining
    mps_pretrain_results = mps_cat.train(mps=mps, loaders=loader,
                                         cfg=cfg.pretrain.mps,
                                         device=device,
                                         title=dataset_name)
    mps = tk.models.MPS(tensors=mps_pretrain_results["best tensors"])

    logger.info("MPS pretraining done.")
