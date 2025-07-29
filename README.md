## Adversarial training with MPS
In this repository, the complete pipeline to improve the generative capabilities of a discriminatively pretrained Matrix Product State (MPS) using GAN-style adversarial training is implemented using `tensorkrowch`, reproducing [Mossi et al. 2024](https://arxiv.org/abs/2406.17441). This is not to be confused with the defense method "adversarial training" where a classifier is trained on adversarial examples alongside normal data to improve robustness against adversarial attacks. 

The first implementation uses 2D toy datasets and others whose generation / download scripts are in `datasets/`. The scripts perform preprocessing that needs to be accessed somehow. The data itself will be stored only locally in a folder called `data/`. 

There are multiple components to adversarial training: 
- Supervised classification using the MPS. `mps_pretraining.py`.
- Sampling using the MPS `sampling.py` (with a custom Born machine distribution function). This is the core of the repository. 
- In the classical adversarial training scheme: An independent, pretrained discriminator `discriminator_pretraining.py`.
- The adversarial training script itself. `adversarial_training.py`.

Experiments are conducted in the folder `experiments/`. Exploratory notebooks are stored only locally. Dependencies are captured in the `environment.yml` file. 

