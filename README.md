# Adversarial Training with MPS

## Study
This repository implements the complete pipeline to improve the generative capabilities of a discriminatively pretrained **Born Machine** based on **Matrix Product States (MPS)** using GAN-style adversarial training with [tensorkrowch](https://joserapa98.github.io/tensorkrowch/_build/html/index.html), reproducing [Mossi et al., 2024](https://arxiv.org/abs/2406.17441).

The goal is to scale from simple 2D toy datasets to more complex data where adversarial attacks are meaningful. This will be followed by a study of the **robustness of different models against adversarial examples**. The models to compare include:

- Discriminatively trained MPS  
- MPS after GAN-style training  
- MPS after adversarial training (MPS as the discriminator)  
- Benchmark against state-of-the-art neural network-based models

---

## Requirements
The easiest setup is via a custom conda environment. From the parent folder, run:

```bash
conda env create -f environment.yml
```
All required packages are listed in `environment.yml.`

---

## Repository Structure
The repository is organized as follows:
### 1. Source code (`src/`)
Implements:
    - Custom inference and sampling algorithms
    - data preprocessing, and
    - Training steps for different models and paradigms

### 2. Experiment scripts (`experiments/`)
Run experiments as modules from the parent folder: 
```bash 
python -m experiments.<script>
```

### 3. Configuration files (`configs/`)
Experiment configurations are modular and managed with [Hydra](https://hydra.cc/). 
Configs for individual runs or multiruns are stored in `configs/experiments`. 
Single run:
```bash 
python -m experiments.<script> +experiments=<config>
```
Multirun:
```bash 
python -m experiments.<script> --multirun +experiments=<config>
```

---

## Tracking and Outputs
Runs are tracked by [Weights & Biases](https://wandb.ai/).
Local folders created during experiments (not tracked by Git) include:
- `.datasets/`: Downloaded/generated datasets,
- `outputs/`: Single run config logs and model weights
- `multirun/`: Multirun congig logs and model weights
- `wandb/`: Run history and figures.
