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
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()
