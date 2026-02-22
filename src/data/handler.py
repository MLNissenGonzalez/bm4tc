from src.models import BornMachine
import torch
import src.utils.schemas as schemas
from src.data.gen_n_load import load_dataset, LabelledDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import *
from torch.utils.data import DataLoader, TensorDataset
import src.utils.get as get
from math import ceil
import logging

logger = logging.getLogger(__name__)

_SCALER_MAP = {
    "minmax": MinMaxScaler
}


class DataHandler:
    """
    Central data management for loading, preprocessing, and providing data loaders.

    Handles dataset loading, train/valid/test splitting, feature scaling,
    and provides DataLoaders for both classification and GAN-style training.

    Attributes:
        data: Dict of split tensors {"train", "valid", "test"} after preprocessing.
        labels: Dict of label tensors {"train", "valid", "test"}.
        data_dim: Number of input features.
        num_cls: Number of classes.
        classification: DataLoaders for classification training.
        discrimination: DataLoaders for GAN-style training (samples per class).
    """

    def __init__(self, cfg: schemas.DatasetConfig):
        """
        Initialize DataHandler with dataset configuration.

        Args:
            cfg: Dataset configuration with gen_dow_kwargs, split ratios, etc.
        """
        self.cfg = cfg
        self.data = None
        self.labels = None
        self.means: List[torch.Tensor] = None
        self.covs: List[torch.Tensor] = None
        self.classification: Dict[str, DataLoader] = None
        self.discrimination: Dict[str, DataLoader] = None
        self.data_dim: int = None
        self.num_cls: int = None
        self.total_size: int = None
        self.num_spc: List[int] = None

    def load(self):
        """Load raw dataset from disk (or generate if needed)."""
        lbld_data = load_dataset(self.cfg)
        self.data, self.labels = lbld_data.X, lbld_data.t
        self.data_dim, self.num_cls, self.total_size = lbld_data.num_feat, lbld_data.num_cls, lbld_data.size
        

    def _compute_mean_and_covariance(self, data: torch.FloatTensor):
        mean = data.mean(dim=0)
        cov = torch.cov(data.T)
        return mean, cov

    def _get_scaler(self, scaler_name: str):
        key = scaler_name.lower().replace("-", "").replace(" ", "")
        scaler_cls = _SCALER_MAP.get(key)
        if scaler_cls is None:
            raise ValueError(f"Unknown scaler '{scaler_name}'")
        self.scaler = scaler_cls(feature_range=self.input_range, clip=True)


    def split_and_rescale(
            self,
            bornmachine: BornMachine,
            scaler_name: str = "minmax"
    ):
        """
        Split data into train/valid/test and rescale to embedding range.

        Args:
            bornmachine: BornMachine (used to determine input range from embedding).
            scaler_name: Scaler type ("minmax").
        """
        # Check that data is already loaded to handler, otherwise, load it:
        if self.data is None:
            self.load(self.cfg)
        # Final input range depends on the embedding chosen for the Born-Machine.    
        embedding_name = bornmachine.embedding_name
        self.input_range = get.range_from_embedding(embedding_name)
        # Initalize dictionaries
        data = {}
        labels = {}
        # First split: separate test set
        split_ratios = self.cfg.split
        (remaining_data, data["test"], 
         remaining_labels, labels["test"]) = train_test_split(
                                            self.data, self.labels, 
                                            test_size=split_ratios[-1], 
                                            random_state=self.cfg.split_seed
                                        )
        # Second split: separate validation set from remaining data
        # ratio_new = ratio_old / ratio_remaining
        (data["train"], data["valid"], 
         labels["train"], labels["valid"]) = train_test_split(
                                            remaining_data, remaining_labels, 
                                            test_size=split_ratios[1]/(1-split_ratios[-1]), 
                                            random_state=self.cfg.split_seed
                                        )
        # Fit scaler to training data to avoid data leakage.
        self._get_scaler(scaler_name)
        data["train"] = self.scaler.fit_transform(data["train"])
        # Transform validation and test sets using the scaler fitted on training data
        data["valid"] = self.scaler.transform(data["valid"])
        data["test"] = self.scaler.transform(data["test"])
        # Convert to PyTorch tensors for subsequent training
        self.data, self.labels = {}, {}
        for split in ["train", "valid", "test"]:
            self.data[split] = torch.FloatTensor(data[split])
            self.labels[split] = torch.LongTensor(labels[split])

        # If not to expensive, compute dataset statistics
        all_data = torch.cat([d for d in self.data.values()], dim=0)
        all_labels = torch.cat([l for l in self.labels.values()])
        self.classified_data = []
        if self.data_dim < 1e2:
            self.means, self.covs = [], []
        for c in range(self.num_cls):
            class_data = all_data[all_labels == c]
            self.classified_data.append(class_data)
            if self.data_dim < 1e2:
                mean, cov = self._compute_mean_and_covariance(class_data)
                self.means.append(mean), self.covs.append(cov)
        self.classified_data = torch.stack(self.classified_data, dim=1) # already more or less sorted into splits
        del all_data
        del all_labels

    def get_classification_loaders(self, batch_size: int = 64):
        """Create DataLoaders for classification training (data, labels) pairs.

        Args:
            batch_size: Number of samples per batch. Should match the trainer's
                        configured batch_size (from ClassificationConfig, etc.).
        """
        if not isinstance(self.data, dict):
            raise AttributeError(f"Call split_and_rescale first.")

        self.classification = {}
        for split, split_data in self.data.items():
            split_labels = self.labels[split]
            lbd_data = TensorDataset(split_data, split_labels)
            effective_bs = min(batch_size, len(split_data))
            if effective_bs < batch_size:
                logger.warning(
                    f"batch_size ({batch_size}) exceeds {split} split size "
                    f"({len(split_data)}); clamping to {effective_bs}."
                )
            self.classification[split] = DataLoader(lbd_data,
                                             batch_size=effective_bs,
                                             drop_last=(split=="train"),
                                             shuffle=(split=="train"))
            
    def get_discrimination_loaders(self, batch_size: int = 16):
        """
        Create DataLoaders for GAN-style training.

        Provides batches of shape (batch_size, num_classes, data_dim) for
        comparing real samples against synthetic samples from the generator.

        Args:
            batch_size: Number of samples per class per batch.
        """
        num_spc = self.classified_data.shape[0]

        # Convert ratios to cumulative boundaries
        ratios = self.cfg.split
        assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"

        boundaries = [
            int(round(ratios[0] * num_spc)),
            int(round((ratios[0] + ratios[1]) * num_spc)),
            num_spc
        ] # i.e. the indices indicating where to split the data

        splits = {
            "train": (0, boundaries[0]),
            "valid": (boundaries[0], boundaries[1]),
            "test":  (boundaries[1], boundaries[2]),
        }

        self.discrimination = {}

        for split, (a, b) in splits.items():
            split_tensor = self.classified_data[a:b]  # shape: (num_spc, num_classes, data_dim)

            self.discrimination[split] = DataLoader(
                split_tensor,
                batch_size=batch_size,
                drop_last=(split == "train"),
                shuffle=(split == "train"),
            )

        del self.classified_data
        self.num_spc = [
            int(round(ratios[0] * num_spc)),
            int(round(ratios[1] * num_spc)),
            int(round(ratios[2] * num_spc))
        ]