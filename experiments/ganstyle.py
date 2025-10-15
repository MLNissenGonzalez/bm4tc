import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "src"))  # make src importable


from collections import defaultdict
import logging
from omegaconf import OmegaConf
import numpy as np
import torch
import tensorkrowch as tk
import hydra
import matplotlib.pyplot as plt
from src._utils import init_wandb, _class_wise_dataset_size, visualise_samples, save_model, verify_tensors, set_seed
import src.schemas as schemas
from src.datasets.gen_n_load import load_dataset, LabelledDataset
from src.datasets.preprocess import preprocess_pipeline
import src.mps.categorisation as mps_cat
import src.discriminator.utils as discr
import src.mps.sampling as sampling
import src.gantrain as gantrain
import wandb


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):

    # Initialising wandb and device
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.reproduce.random.seed)
    logger.info(f"{device=}")

    # Loading dataset for classification
    dataset: LabelledDataset = load_dataset(cfg=cfg.dataset)
    dataset_name = dataset.name
    data_dim = dataset.num_feat
    num_cls = dataset.num_cls

    # Initialising MPS
    init_cfg = OmegaConf.to_object(cfg.model.mps.init_kwargs)
    classifier = tk.models.MPSLayer(n_features=data_dim+1,
                             out_dim=num_cls,
                             device=device,
                             **init_cfg)
    cls_pos = classifier.out_features[0]  # important global variable
    classifier.embedding = cfg.model.mps.embedding

    info = f"MPS initalised with:\n {classifier.bond_dim=} and\n{classifier.in_dim=}"
    info_lines = info.split("\n")
    logger.info("\n".join(info_lines))

    # Preprocessing data to be ready for embedding
    X, t, _ = preprocess_pipeline(X_raw=dataset.X, t_raw=dataset.t,
                                       split=cfg.dataset.split,
                                       random_state=cfg.reproduce.random.random_state,
                                       embedding=classifier.embedding)
    
    # Wrapping preprocessed data in loader for later GAN-style training
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
        

    # Visualising the data
    ax = visualise_samples(samples=X["train"], labels=t["train"], gen_viz=cfg.sampling.num_spc)
    wandb.log({"samples/dataset": wandb.Image(ax.figure)})
    plt.close(ax.figure)

    # Load Pretrained MPS
    # TODO: Close gaps
    mps_pretrain_tensors, mps_pretrain_best_acc = torch.load(cfg.model.mps.path)
    classifier.initialize(tensors=mps_pretrain_tensors, device=device)

    # From here on, MPS as generator in the foreground
    generator = tk.models.MPS(
        tensors=mps_pretrain_tensors, device=device)
    verify_tensors(classifier, generator, "Classifier", "Generator")
    generator.embedding = classifier.embedding

    # Save locally and as wandb.artifact, if wanted
    if cfg.reproduce.save.pre_mps:
        path_pre_mps = save_model(
            model=generator, run_name=run.name, model_type="pre_mps")
        run.log_model(path=path_pre_mps, name=f"pre_mps_{run.name}")

    # Preparing dataset for discriminator pretraining
    X_synth = {}
    dis_loaders = {}
    n_spc = {}
    for split in ["train", "valid", "test"]:
        # Sampling fake examples
        n_spc[split] = max(_class_wise_dataset_size(
            t=t[split], num_cls=num_cls))
        logger.debug(f"Amount of samples per class generated = {n_spc}.")
        with torch.no_grad():
            X_synth[split] = (
                sampling.batched(
                    mps=generator, 
                    embedding=generator.embedding,
                    cls_pos=cls_pos,
                    num_spc=n_spc[split],
                    num_bins=cfg.sampling.batch_spc,
                    batch_spc=cfg.sampling.batch_spc,
                    method=cfg.sampling.method,
                    device=device
                )
                .cpu()
            )
        torch.cuda.empty_cache()

        # Wrapping in the loader
        dis_loaders[split] = discr.pretrain_loader(X_real=X[split],
                                                   c_real=t[split],
                                                   X_synth=X_synth[split],
                                                   mode=cfg.model.dis.mode,
                                                   batch_size=cfg.pretrain.dis.batch_size,
                                                   split=split)

    # Initialising discriminator(s)
    d = discr.init_discriminator(
        cfg=cfg.model.dis, input_dim=data_dim, num_classes=num_cls, device=device)
    info = f"Discriminator initalised with:\nhidden multipliers = {cfg.model.dis.hidden_multipliers} and\nactivation = {cfg.model.dis.nonlinearity}"
    info_lines = info.split("\n")
    logger.info("\n".join(info_lines))

    # Swapping dictionary nesting to fit logic below
    d_loaders = defaultdict(dict)
    for split, dic in dis_loaders.items():
        for i, loader in dic.items():
            d_loaders[i][split] = loader
    logger.info("Data for pretraining of discriminator loaded.")

    # Vizualising generative capabilities after pretraining
    to_visualise = X_synth.get("train")
    ax = visualise_samples(samples=to_visualise, labels=None, gen_viz=cfg.sampling.batch_spc)
    wandb.log({"samples/pre": wandb.Image(ax.figure)})
    plt.close(ax.figure)

    # Discriminator pretraining
    for i in d.keys():
        d[i] = discr.pretraining(
            dis=d[i],
            cfg=cfg.pretrain.dis,
            loaders=d_loaders[i],
            key=i,
            device=device
        )
        # Save locally and as wandb.artifact if wanted
        if cfg.reproduce.save.pre_dis:
            path_pre_dis = save_model(
                model=d[i], run_name=f"{i}_{run.name}", model_type="pre_dis")
            run.log_model(path=path_pre_dis, name=f"pre_dis_{i}_{run.name}")
        logger.info(f"Pretraining of discriminator {i} completed.")
    
    logger.info("Pretraining completed.")

    # Constructing dataloader of real, unembedded data
    real_loaders = {}
    n_real = num_cls * cfg.sampling.batch_spc
    for split in ["train", "valid", "test"]:
        real_loaders[split] = gantrain.real_loader(
            X=X[split], c=t[split],
            n_real=n_real, split=split
        )

    # GAN style training
    logger.info("GAN-style training begins.")
    best_acc = mps_pretrain_best_acc
    gantrain.loop(
        generator=generator, dis=d, real_loaders=real_loaders,
        cfg=cfg.gantrain, cls_pos=cls_pos, samp_cfg=cfg.sampling,
        embedding=cfg.model.mps.embedding, best_acc=best_acc, 
        cat_loaders=loaders, device=device
    )
    
    # Save locally and as wandb.artifact, if wanted
    if cfg.reproduce.save.gan_dis:
        for i in d.keys():
            path_gan_dis = save_model(
                model=d[i], run_name=f"{i}_{run.name}", model_type="gan_dis")
            run.log_model(path=path_gan_dis, name=f"gan_dis_{i}_{run.name}")
    if cfg.reproduce.save.gan_mps:
        path_gan_mps = save_model(
            model=generator, run_name=run.name, model_type="gan_mps")
        run.log_model(path=path_gan_mps, name=f"gan_mps_{run.name}")
    
    logger.info("GAN-style training completed.")

    # Visualizing generative capabilities after GAN-style training
    if data_dim == 2:
        n = n_spc["train"]
    else:
        n = cfg.sampling.batch_spc # maybe change this
    with torch.no_grad():
        synths = sampling.batched(
            mps=generator,
            embedding=cfg.model.mps.embedding,
            method=cfg.sampling.method,
            cls_pos=cls_pos, num_spc=n,
            num_bins=cfg.sampling.num_bins,
            batch_spc=cfg.sampling.batch_spc,
            device=device).cpu()
    torch.cuda.empty_cache()
    ax = visualise_samples(samples=synths)
    wandb.log({"samples/gan": wandb.Image(ax.figure)})
    plt.close(ax.figure)

    # Closing the run
    run.finish()


if __name__ == "__main__":
    main()
