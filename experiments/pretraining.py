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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

from collections import defaultdict
import numpy as np
import logging
from omegaconf import OmegaConf
import torch
import tensorkrowch as tk
import hydra

from src._utils import _class_wise_dataset_size
from src.schemas import Config
from src.datasets.gen_n_load import load_dataset, LabelledDataset
from src.datasets.preprocess import preprocess_pipeline
import src.mps.categorisation as mps_cat
import src.discriminator.utils as dis
import src.mps.sampling as sampling

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

    # 6. Discriminator initialization (could be an ensemble here)
    d = dis.init_discriminator(
        cfg=cfg.model.dis, input_dim=data_dim, num_classes=num_cls, device=device)

    # 7. Synthezesing and wrapping to data loader
    X_synth = {}
    dis_loaders = {}
    for split in ["train", "valid", "test"]:
        n_spc = max(_class_wise_dataset_size(t[split]))
        logger.debug(f"Amount of samples per class generated = {n_spc}.")
        X_synth[split] = sampling.batched(
            mps=mps, embedding=cfg.model.mps.embedding,
            cls_pos=cls_pos,
            num_spc=n_spc,
            num_bins=cfg.gantrain.num_bins,
            batch_spc=cfg.gantrain.n_synth,
            device=device
        ).detach()  # We do not want MPS gradients.
        dis_loaders[split] = dis.pretrain_loader(X_real=X[split],
                                                 c_real=t[split],
                                                 X_synth=X_synth[split],
                                                 mode=cfg.model.dis.mode,
                                                 batch_size=cfg.pretrain.dis.batch_size,
                                                 split=split
                                                 )
    d_loaders = defaultdict(dict)
    for split, dic in dis_loaders.items():
        for i, loader in dic.items():
            d_loaders[i][split] = loader
    logger.info("Data for pretraining of discriminator loaded.")

    # 8. Discriminator pretraining
    dis_pretrain_results = {}
    for i in d.keys():
        dis_pretrain_results[i] = dis.pretraining(dis=d[i],
                                                  cfg=cfg.pretrain.dis,
                                                  loaders=d_loaders[i])
        logger.info(f"Pretraining of discriminator {i} completed.")
    logger.info("Pretraining completed.")


if __name__ == "__main__":
    main()
