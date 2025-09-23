import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable

from collections import defaultdict
import logging
from omegaconf import OmegaConf
import torch
import tensorkrowch as tk
import hydra
import matplotlib.pyplot as plt
from src._utils import _class_wise_dataset_size, visualise_samples
import src.schemas as schemas
from src.datasets.gen_n_load import load_dataset, LabelledDataset
from src.datasets.preprocess import preprocess_pipeline
import src.mps.categorisation as mps_cat
import src.discriminator.utils as dis
import src.mps.sampling as sampling
import src.gantrain as gantrain
import wandb



logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):

    # Initialising wandb and device
    run = schemas.init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device=}")

    # Loading dataset for classification
    dataset: LabelledDataset = load_dataset(cfg=cfg.dataset)
    dataset_name = dataset.name
    data_dim = dataset.num_feat
    num_cls = dataset.num_cls

    # Initialising MPS
    init_cfg = OmegaConf.to_object(cfg.model.mps.init_kwargs)
    mps = tk.models.MPSLayer(n_features=data_dim+1,
                             out_dim=num_cls,
                             device=device,
                             **init_cfg)
    cls_pos = mps.out_features[0]  # important global variable

    # Preprocessing data to be ready for embedding
    X, t, scaler = preprocess_pipeline(X_raw=dataset.X, t_raw=dataset.t,
                                       split=cfg.dataset.split,
                                       random_state=cfg.dataset.split_seed,
                                       embedding=cfg.model.mps.embedding)

    # Preparing dataloader for classification via MPS
    loaders = {}
    size_per_class = {}
    for split in ["train", "valid", "test"]:
        loaders[split] = mps_cat.loader_creator(X=X[split],
                                                t=t[split],
                                                batch_size=cfg.pretrain.mps.batch_size,
                                                embedding=cfg.model.mps.embedding,
                                                phys_dim=cfg.model.mps.init_kwargs.in_dim,
                                                split=split)
        size_per_class[split] = _class_wise_dataset_size(t[split], num_cls)
        logging.debug(f"{size_per_class[split]=}")

    # Pretraining the MPS as classifier
    mps_pretrain_tensors, mps_pretrain_best_acc = mps_cat.train(mps=mps, loaders=loaders,
                                                               cfg=cfg.pretrain.mps,
                                                               device=device,
                                                               title=dataset_name,
                                                               stage="pre")
    mps = tk.models.MPS(
        tensors=mps_pretrain_tensors, device=device)
    logger.info("MPS pretraining done.")

    # Initialising discriminator(s)
    d = dis.init_discriminator(
        cfg=cfg.model.dis, input_dim=data_dim, num_classes=num_cls, device=device)

    # Preparing dataset for discriminator pretraining
    X_synth = {}
    dis_loaders = {}
    n_spc = {}
    for split in ["train", "valid", "test"]:
        # Sampling fake examples
        n_spc[split] = max(_class_wise_dataset_size(t=t[split], num_cls=num_cls))
        logger.info(f"Amount of samples per class generated = {n_spc}.")
        X_synth[split] = sampling.batched(
            mps=mps, embedding=cfg.model.mps.embedding,
            cls_pos=cls_pos,
            num_spc=n_spc[split],
            num_bins=cfg.gantrain.num_bins,
            batch_spc=cfg.gantrain.n_real,
            device=device).detach() 
        # Wrapping in the loader
        dis_loaders[split] = dis.pretrain_loader(X_real=X[split],
                                                 c_real=t[split],
                                                 X_synth=X_synth[split],
                                                 mode=cfg.model.dis.mode,
                                                 batch_size=cfg.pretrain.dis.batch_size,
                                                 split=split)
        
    # Swapping dictionary nesting to fit logic below
    d_loaders = defaultdict(dict)
    for split, dic in dis_loaders.items():
        for i, loader in dic.items():
            d_loaders[i][split] = loader
    logger.info("Data for pretraining of discriminator loaded.")

    # Vizualising generative capabilities after pretraining     
    to_visualise = X_synth.get("train")
    ax = visualise_samples(samples=to_visualise, labels=None, gen_viz=cfg.wandb.gen_viz)
    wandb.log({"samples/pretraining": wandb.Image(ax.figure)})
    plt.close(ax.figure)

    # Discriminator pretraining
    for i in d.keys():
        d[i] = dis.pretraining(
            dis=d[i],
            cfg=cfg.pretrain.dis,
            loaders=d_loaders[i],
            key=i,
            device=device
        )
        logger.info(f"Pretraining of discriminator {i} completed.")
    logger.info("Pretraining completed.")

    # Constructing dataloader of real, unembedded data
    real_loaders = {}
    for split in ["train", "valid", "test"]:
        real_loaders[split] = gantrain.real_loader(
            X=X[split], c=t[split],
            n_real=cfg.gantrain.n_real, split=split
        )

    # GAN style training
    logger.info("GAN-style training begins.")
    best_acc = mps_pretrain_best_acc
    mps = gantrain.loop(
        mps=mps, dis=d, real_loaders=real_loaders,
        cfg=cfg.gantrain, cls_pos=cls_pos,
        embedding=cfg.model.mps.embedding,
        best_acc=best_acc, cat_loaders=loaders,
        device=device
    )
    logger.info("GAN-style training completed.")

    # Visualizing generative capabilities after GAN-style training
    if data_dim == 2:
        n = n_spc["train"]
    else: 
        n = cfg.wandb.gen_viz
    synths = sampling.batched(
            mps=mps, embedding=cfg.model.mps.embedding,
            cls_pos=cls_pos,
            num_spc=n,
            num_bins=cfg.gantrain.num_bins,
            batch_spc=cfg.gantrain.n_real,
            device=device).detach()
    ax = visualise_samples(samples=synths)
    wandb.log({"samples/gantraining": wandb.Image(ax.figure)})
    plt.close(ax.figure)


    run.finish()

if __name__ == "__main__":
    main()
