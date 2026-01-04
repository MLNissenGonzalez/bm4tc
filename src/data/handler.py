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

_SCALER_MAP = {
    "minmax": MinMaxScaler
}
class DataHandler:
    def __init__(self, cfg: schemas.DatasetConfig):
        self.cfg = cfg
        self.data = None
        self.labels = None
        self.means : List[torch.Tensor] = None
        self.covs : List[torch.Tensor] = None
        self.classification : Dict[str, DataLoader] = None
        self.discrimination: Dict[str, DataLoader] = None
        self.data_dim : int = None 
        self.num_cls : int = None 
        self.total_size : int = None
        self.num_spc: List[int] = None

    def load(self):
        lbld_data = load_dataset(self.cfg)
        self.data, self.labels = lbld_data.X, lbld_data.t
        self.data_dim, self.num_cls, self.total_size = lbld_data.num_feat, lbld_data.num_cls, lbld_data.size
        # load data to Handler. use gen_n_load script.
        

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

    def get_classification_loaders(self):
        if not isinstance(self.data, dict):
            raise AttributeError(f"Call split_and_rescale first.")
        
        self.classification = {}
        for split, split_data in self.data.items():
            split_labels = self.labels[split]
            lbd_data = TensorDataset(split_data, split_labels)
            self.classification[split] = DataLoader(lbd_data, 
                                             batch_size=64, 
                                             drop_last=(split=="train"),
                                             shuffle=(split=="train"))
            
    def get_discrimination_loaders(self):

        num_spc = self.classified_data.shape[0]

        # Convert ratios to cumulative boundaries
        ratios = self.cfg.split
        assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"

        boundaries = [
            int(round(ratios[0] * num_spc)),
            int(round((ratios[0] + ratios[1]) * num_spc)),
            num_spc
        ]

        splits = {
            "train": (0, boundaries[0]),
            "valid": (boundaries[0], boundaries[1]),
            "test":  (boundaries[1], boundaries[2]),
        }

        self.discrimination = {}

        for split, (a, b) in splits.items():
            split_tensor = self.classified_data[a:b]  # shape: (num_spc, num_classes, data_dim)

            dataset = TensorDataset(split_tensor)

            self.discrimination[split] = DataLoader(
                dataset,
                batch_size=16,
                drop_last=(split == "train"),
                shuffle=(split == "train"),
            )

        del self.classified_data
        self.num_spc = [
            int(round(ratios[0] * num_spc)),
            int(round(ratios[1] * num_spc)),
            int(round(ratios[2] * num_spc))
        ]