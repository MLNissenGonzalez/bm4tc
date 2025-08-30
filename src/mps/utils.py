import torch
import tensorkrowch as tk
from typing import Union, Sequence, Callable, Dict
import src.sampling as sampling
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from schemas import PretrainMPSConfig
from _utils import get_optimizer, get_criterion

# TODO: Think about splitting utils into utils and pretraining. 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------Born rule. Sequential and parallel code.------------------------------------------------------------------------------------------------------------------
#------Could add this maybe as method in a custom MPS class.--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------torch.optim.Optimizer------------------------------------------------------------

# TODO: batch_size could be a confusing misnomer in the documentation. batch_size could be the product of num_bins and num_dist one want so compute.
def born_sequential(mps: tk.models.MPS, 
                    embs: dict | torch.Tensor)-> torch.Tensor: # very flexible, not parallizable
    """ 
    Sequential contraction of MPS with embedded input and computation of probabilities using the Born rule. Mainly used for sampling.

    Cases
    -----
    - Not all variables (including class index) appear in p({x_i}_I|{x_j}_J), i.e. in embs. 
        In that case, one has to marginalize over the missing variables resulting in
        a density matrix whose diagonal contrains the unnormalized distribution(s) (batch_size x num_bins values). 
    - All variables appear in embs, one has to square the output of the mps to obtain the unnormalized distribution(s). 
    
    Parameters
    ----------
    mps:    tk.models.MPS
        instance defining the prob amplitude
    embs:   dict
        embedded input as dict of batch_size x phys_dim tensors  with keys indicating input position.
    
    Returns
    -------
    tensor
        probability distribution(s)
    """
    
    # TODO: Maybe remove this check, as it might slow the code down.
    if not isinstance(embs, dict):
        raise TypeError("embs input needs to be dictionary with keys indicating input position and values embedded input feature.")
    
    # Advantage of dictionaries is that one can use the keys to save which legs are input and which are not. 
    mps.in_features = [i for i in embs.keys()]
    # tensorkrowch processes lists of inputs sequentially.
    in_tensors = [embs[i] for i in mps.in_features]
    
    # Case 1: Not all variables appear, thus marginalize_output=True and one has to take the diagonal.
    if len(in_tensors) < mps.n_features:
        return torch.diagonal(mps(in_tensors, marginalize_output=True))
    # Case 2: All variables appear.
    else:
        return torch.square(mps.forward(data=in_tensors))
    
def born_parallel(mps: tk.models.MPSLayer | tk.models.MPS, # could use MPSLayer class for this one actually
                  embs: torch.Tensor)-> torch.Tensor:
    """ 
    Parallel contraction of MPS with embedded input and computation of probabilities using the Born rule. Mainy used for classification.

    Cases
    -----
    - If mps has n_features = D (only input sites):
        Returns joint Born probabilities p(x₁,…,x_D) ∝ |ψ(x)|².
        Shape: (batch_size, 1)

    - If mps has n_features = D+1 (input sites + output site):
        Returns conditional class probabilities p(c|x₁,…,x_D).
        Shape: (batch_size, num_cls)

    Parameters
    ----------
    mps : tk.models.MPSLayer | tk.models.MPS
        MPS model defining the probability amplitude.
    embs : torch.Tensor
        Embedded input of shape (batch_size, D, phys_dim).

    Returns
    -------
    torch.Tensor
        Probabilities with shape (batch_size, num_cls).
    """
    
    # TODO: Think about removing this check for efficiency. 
    if not isinstance(embs, torch.Tensor): # embs=tensor (parallizable), assume mps.out_feature = [cls_pos] globally??
        raise TypeError("embs input needs to be tensor of shape: (batch_size, D, phys_dim)")
    
    is_joint = isinstance(mps, tk.models.MPS) and (mps.n_features == embs.shape[1])
    # Case: p(c|x_1,..., x_D), since n_features = D+1 (with output site)
    if not is_joint:
        p = torch.square(mps.forward(data=embs)) 
        return p / p.sum(dim=1, keepdim=True) 
    # Case: p(x_1,...,x_D), since n_features = D (no central tensor).
    else:
        return torch.square(mps.forward(data=embs)).unsqueeze(1) # Z x p(x_1,..., x_D) # shape (batch_size, 1)
    

# TODO: Think of deleting this function as it may never be used.
def batch_normalize(p: torch.Tensor,
                    num_bins: int,
                    batch_size: int):
    """
    Normalize a batch of unnormalized discretized (univariate) probability distributions stored in a single tensor.
    
    Parameters
    ----------
    p: tensor
        batch of unnormalized probability distributions stored in a single tensor of shape (batch_size, num_bins) or (batch_size x num_bins).
    batch_size: int
        number of unnormalized probability distributions stored in the tensor p
    num_bins: int
        number of discretization points for every single unnormalized probability distribution p[i, :].

    Returns
    -------
    tensor
        single tensor storing batch_size many normalized univariate discrete probability distributions of num_bins bins.
    """
    if p.shape == (batch_size*num_bins,):
        p = p.reshape(batch_size, num_bins)

    return p / p.sum(dim=1, keepdim=True)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Sampling routines using mps------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Currenty on purely sequential implementation-------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: Interface or implementation level?
# TODO: Think about moving away from dictionaries
# TODO: Add other sampling methods and add it as configuration for experiments
# TODO: Simplify code assuming:
#       - same embedding at all input sites (this excludes the class site).
#       - same number of bins for all input sites
def _cc_mps_sampling(mps: tk.models.MPS,
                    input_space: Union[torch.Tensor, Sequence[torch.Tensor]], # need to have the same number of bins
                    embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                    num_samples: int, # per class
                    cls_pos: int, # extract this once for the mps, not at every sampling step
                    cls_emb: torch.Tensor, # perform embedding once, torch.Size=[num_cls]
                    device: torch.device
                    )-> torch.Tensor:
    """
    This samples num_samples samples per class.

    Parameters
    ----------
    mps: MPS instance
        mps to sample from
    input_space: list of tensors
        space to sample into. 
    embedding: 
        mapping input dimension to larger embedding dimension
    num_samples: int
        number of samples per class
    cls_pos: int
        which position in the mps corresponds to the class output assuming a central tensor mps design
    cls_emb: tensor
        the basis embedding of a single class, just a one-hot vector in most cases
    device: torch device

    Returns
    -------
    tensor
        num_samples samples with dims num
    """
    if isinstance(input_space, torch.Tensor):
        input_space = [input_space.clone() for _ in range(mps.n_features - 1)]
    elif len(input_space) != (mps.n_features-1):
        raise ValueError(f'Expected {mps.n_features-1} input spaces, got {len(input_space)}')
    
    if isinstance(embedding, Callable):
        embedding = [embedding] * (mps.n_features-1)
    elif len(embedding) != (mps.n_features-1):
        raise ValueError(f'Expected {mps.n_features-1} embedding functions, got {len(embedding)}')
    
    phys_dim = mps.phys_dim[:cls_pos] + mps.phys_dim[cls_pos+1:]
    embs = {}
    samples = {}

    num_bins = input_space[0].shape[0] # Assumes same batch_size for input feature
    embs[cls_pos] = cls_emb[None, :].expand(num_samples * num_bins, -1).to(device)
    site = 0
    for i in range(mps.n_features):
        if i == cls_pos:
            continue   

        num_bins = input_space[site].shape[0]
        embedding_out = embedding[site](input_space[site], phys_dim[site])  # [num_bins, phys_dim]
        expanded = embedding_out[None, :, :].expand(num_samples, -1, -1)  # [num_samples, num_bins, phys_dim]
        embs[i] = expanded.reshape(num_samples * num_bins, phys_dim[site]).to(device) # [num_samples * num_bins, phys_dim]

        mps.unset_data_nodes()
        mps.reset()
        mps.to(device)
        
        p = born_sequential(mps=mps, embs=embs).view(num_samples, num_bins) # [num_samples, num_bins]
        
        # Later, include an if clause here to inclue other sampling methods, or make this a configuration
        samples[site] = sampling.sss_sampling(p, input_space[site])
        
        embs[i] = embedding[site](samples[site], phys_dim[site])[:, None, :].expand(-1, num_bins, -1).reshape(num_samples*num_bins, -1)
        site += 1
    return samples

# TODO: Think about moving away from dictionaries
def mps_sampling(   mps: tk.models.MPS,
                    input_space: Union[torch.Tensor, Sequence[torch.Tensor]], # need to have the same number of bins
                    embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                    num_samples: int, # per class
                    cls_pos: int,
                    cls_embs: Sequence[torch.Tensor], # one embedding per class,
                    device: torch.device) -> torch.Tensor:
    """
    Samples num_samples samples for each class. Basically just a loop around _cc_mps_sampling.

    Parameters
    ----------
    mps: MPS model
    input_space: tensor
    embedding: tk.embedding
    num_samples: int
        number of samples per class
    cls_pos: int
        position of central tensor
    cls_embs: list of tensors
        precomputed class embeddings for each class
    device: torch.device

    Returns
    -------
    tensor
        (num_classes x num_samples) number of samples in input space (not embedded)
    """
    
    samples = []
    for cls_emb in cls_embs:
        cls_samples = _cc_mps_sampling(
            mps=mps,
            input_space=input_space,
            embedding=embedding,
            num_samples=num_samples,
            cls_pos=cls_pos,
            cls_emb=cls_emb,
            device=device
        )
        samples = samples + [torch.stack(list(cls_samples.values()), dim=1)] # num_samples x (n_features-1), where n_features counts class as feature

    samples = torch.concat(samples) # (num_classes x num_samples) x (n_features-1), ordered by class for later retrieval, if wanted
    return samples 

# TODO: Add code to discard surplus of samples
# TODO: Consider parallel implementation
def batch_sampling_mps( mps: tk.models.MPS,
                        input_space: Union[torch.Tensor, Sequence[torch.Tensor]], # need to have the same number of bins
                        embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                        batch_size: int, # will be num_samples in mps_sampling
                        cls_pos: int,
                        cls_embs: Sequence[torch.Tensor], # one embedding per class,
                        total_n_samples: int, # total number of samples per class, ought to be divisible by batch_size? hard coding is bad
                        device: torch.device,
                        save_classes = False, # only needed for visualisation
):
    """
    Dividing up sampling of many samples into smaller batches for more economic memory usage.
    Steps in the loop don't depend on each other -> Parallization possible.

    Parameters
    ----------
    mps: MPS model
    input_space: tensor
    embedding: tk.embeddings
        mapping input feature to larger physical dimension
    batch_size: int
        number of samples to be sampled per (mini)batch per class
    cls_pos: int
        position of central tensor in mps
    cls_embs: list of tensors
        precomputed class embeddings (one hot vectors)
    total_n_samples: int
        total number of samples to be sampled per class
    device: torch.device
    save_classes: Boolean
        If true, class labels are saved and returned.
    
    Returns
    -------
    tensor
        samples
    """
    num_batches = (total_n_samples + batch_size - 1) // batch_size
    labels = []
    samples = []
    for _ in range(num_batches): # could compute these batches in parallel, not to bad
        batch = mps_sampling(   mps=mps,
                                input_space=input_space,
                                embedding=embedding,
                                num_samples=batch_size,
                                cls_pos=cls_pos,
                                cls_embs=cls_embs,
                                device=device
            
        ) # (batch_size x num_cls, data.dim) samples
        samples = samples + [batch]
        if save_classes == True:
            batch_label = torch.concat([torch.full((batch_size,), cls, dtype=torch.long) for cls in range(len(cls_embs))])
            labels = labels + [batch_label]
    
    samples = torch.concat(samples) # (num_batches x num_cls, data.dim) samples (I would have to clip this such that there aonly total_n_samples x num_classes)
    if save_classes == True:
        labels = torch.concat(labels)
        return samples, labels
    else:
        return samples

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------MPS usage as a classifier-------------------------------------------------------------------------------------------------------------------------------------
#------Functions and training steps----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mps_cat_loader( X: torch.Tensor, 
                    t: torch.Tensor, 
                    embedding: Callable[[torch.Tensor, int], torch.Tensor], 
                    batch_size: int, 
                    phys_dim: int,
                    split: str) -> DataLoader:
    """
    Create DataLoaders for multiple datasets and splits

    Parameters
    ----------
    X: torch.Tensor, shape: (batch_size, n_feat)
        The preprocessed, non-embedded features.
    t: torch.Tensor, shape (batch_size,)
        The labels of the features X.
    batch_size: int
        The number of examples of features for all classes (not per class!)
    embedding: tk.embeddings.embedding
        Embedding for features X.
    phys_dim: int, 
        Dimension of embedding space

    Returns
    -------
    DataLoader
        Dataloader for supervised classification.
    """
    X = embedding(X, phys_dim)
    dataset = TensorDataset(X, t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    return loader

def mps_acc_eval(   mps: tk.models.MPS | tk.models.MPSLayer, # relies on mps.out_features = [cls_pos]
                    loader: DataLoader,
                    device: torch.device) -> float:
    """
    Computes the prediction accuracy of a MPS Born machine on a dataset.

    Parameters
    ----------
    mps: MPS model with central tensor
    loader: DataLoader
    device: torch.device

    Returns
    -------
    float
        prediction accuracy
    """
    mps.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, t in loader:
            X, t = X.to(device), t.to(device)
            p = born_parallel(mps, X)
            _, preds = torch.max(p, 1)

            # Counting correct predictions
            correct += (preds == t).sum().item()
            total += t.size(0)
    acc = correct / total
    return acc

# TODO: Think about dynamically implemnting loss function here versus in the whole training loop
def _mps_cls_train_step(mps: tk.models.MPS | tk.models.MPSLayer, # expect mps.out_features = [cls_pos]
                       loader: DataLoader,
                       device: torch.device,
                       optimizer: torch.optim.Optimizer,
                       loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] # expects score, not probs. switch for str parameter
                       ) -> list:
    """
    Single epoch classification training. Returns minibatch-wise train_loss.

    Parameters
    ----------
    mps: MPS model with central tensor
    loader: DataLoader
    optimizer: Optimizer
    loss_fn: LossFunction

    Returns
    -------
    list of floats
        minibatch-wise trainloss 
    """
    train_loss = []
    mps.train()
    for X, t in loader:
        X, t = X.to(device), t.to(device)

        # Compute probs and loss
        p = born_parallel(mps, X)
        loss = loss_fn(p, t)

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

# TODO: Think about logging different or more quantities
# TODO: Consider removing title parameter
def disr_train_mps( mps: tk.models.MPS | tk.models.MPSLayer,
                    loaders: Dict[str, DataLoader], #loads embedded inputs and labels
                    cfg: PretrainMPSConfig,
                    cls_pos: int,
                    phys_dim: int,
                    device: torch.device,
                    title: str | None = None,
                    ):
    """
    Full classification loop for MPS with early stopping based on validation accuracy.

    The training loop uses patience stopping and optional goal accuracy stopping. 
    Loss and optimizer are instantiated from the provided config objects. 
    Model-specific flags (`auto_stack`, `auto_unbind`) and device placement are set automatically.

    Parameters
    ----------
    mps : tk.models.MPS or tk.models.MPSLayer
        The MPS model to train.
    loaders : dict[str, DataLoader]
        Dictionary containing 'train' and 'valid' DataLoaders with embedded inputs and labels.
    cfg : PretrainMPSConfig
        Configuration object containing all training hyperparameters and sub-configs:
        
        Fields:
        - optimizer_cfg : OptimizerConfig
            Name and kwargs for optimizer, e.g., {"name": "adam", "optimizer_kwargs": {"lr": 1e-4}}
        - criterion_cfg : CriterionConfig
            Name and kwargs for loss function, e.g., {"name": "nlll", "kwargs": {"eps": 1e-12}}
        - max_epochs : int
            Maximum number of training epochs.
        - patience : int
            Number of epochs without improvement on validation accuracy before stopping.
        - auto_stack : bool
        - auto_unbind : bool
        - goal_acc : Optional[float]
            If provided, training stops early once validation accuracy reaches this value.
        - print_early_stop : bool
            If True, prints early stopping information.
        - print_updates : bool
            If True, prints epoch-wise validation accuracy.
    cls_pos : int
        Position of the central/output tensor in the MPS.
    device : torch.device
        Device to place model and inputs on (CPU or GPU).
    title : str, optional
        Optional title for printing progress messages.

    Returns
    -------
    best_tensors : list[torch.Tensor]
        The tensors of the MPS corresponding to the best validation accuracy.
    train_loss : list[float]
        Mini-batch-wise training loss collected during training.
    val_accuracy : list[float]
        Epoch-wise validation accuracy.
    """
    val_accuracy = []
    train_loss = []
    best_acc = 0.0
    best_tensors = []  # Container for best model parameters

    # Prepare MPS for training
    mps.to(device)
    mps.auto_stack = cfg.auto_stack
    mps.auto_unbind = cfg.auto_unbind
    mps.unset_data_nodes()
    mps.reset()
    # TODO: Check whether both MPS and MPSLayer instances work here. If not, check or just assume MPSLayer (straight forward).
    mps.out_features = [cls_pos] 
    phys_dim = mps.phys_dim[cls_pos-1] 
    mps.trace(torch.zeros(1, len(mps.in_features), phys_dim).to(device))
        
    # Instantiate criterion and optimizer
    criterion = get_criterion(cfg.criterion_cfg)
    optimizer = get_optimizer(mps.parameters(), cfg.optimizer_cfg)

    patience_counter = 0
    for epoch in range(cfg.max_epochs):
        # Training step
        train_loss = train_loss + _mps_cls_train_step(mps, loaders['train'], 
                                  device, optimizer, criterion)

        # Validation step
        acc = mps_acc_eval(mps, loaders['valid'], device)
        val_accuracy.append(acc)

        # Progress tracking and best model update
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_tensors = mps.tensors
        else:
            patience_counter += 1

        # Early stopping via patience
        if patience_counter > cfg.patience:
            if cfg.print_early_stop:
                print(f"Early stopping via patience at epoch {epoch}")
            break
        # Optional early stopping via goal accuracy
        if (cfg.goal_acc is not None) and (cfg.goal_acc < best_acc):
            if cfg.print_early_stop:
                print(f"Early stopping via goal acc at epoch {epoch}")
            break
        # Update report
        if (epoch == 0) and (title is not None):
            print(f'\nTraining of {title} dataset:\n')
        elif cfg.print_updates and ((epoch+1) % 10 == 0):
            print(f"Epoch {epoch+1}: val_accuracy={acc:.4f}")

    return best_tensors, train_loss, val_accuracy



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------Other mps methods-------------------------------------------------------------------------------------------------------------------------------------
#------May or may not be used----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Write a function that for a given MPS (Layer) returns class embeddings

# TODO: Write a function that gives input potential form of input embeddings, what? 
#       I could write a function that takes an input unprocessed dataset and an embedding and returns preprocessed data (may, or may not be embedded)


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------CONDITIONING VIA NEW CLASS---------------------------------------------------------------------------------------------
#--------I MAY SWITCH TO THIS METHOD IF IT IS BETTER---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


# TODO: Add documentation

# The only case where the class below runs with gradients flowing to R_tensor and
# Cls_tensor is if parametrize=False and inline_input=True for every forward call

class CC_MPS(tk.models.MPS):
    def __init__(self, mps: tk.models.MPS, 
                 cls_pos: int,
                 parametrize: bool):
        # Number of features of the new MPS is reduced by one
        n_feat_new = mps.n_features - 1
        phys_dim = mps.phys_dim[0]
        bond_dim = mps.bond_dim[0]
        self.num_cls = mps.phys_dim[cls_pos]

        # Initialize model with right properties
        super().__init__(n_features=n_feat_new, 
                         phys_dim=phys_dim, 
                         bond_dim=bond_dim, 
                         boundary='obc')

        # Extract and save special tensors as parameters
        assert all(isinstance(mps.mats_env[i].tensor, torch.Tensor) 
                   for i in [cls_pos, cls_pos + 1])

        cls_tensor = mps.mats_env[cls_pos].tensor
        R_tensor = mps.mats_env[cls_pos + 1].tensor

        assert cls_tensor is not None
        assert R_tensor is not None
        self.cls_tensor = nn.Parameter(cls_tensor.clone())
        self.R_tensor = nn.Parameter(R_tensor.clone())

        # Copy all tensors into new mps except special tensor. 
        # Empy node is initialised randomly which is not important
        for old_idx in range(mps.n_features):
            if old_idx == cls_pos or old_idx == cls_pos + 1:
                continue  # skip special cls and R tensors

            # Compute correct new index in the reduced model
            new_idx = old_idx if (old_idx < cls_pos) else (old_idx - 1)
            old_ts = mps.mats_env[old_idx].tensor
            assert old_ts is not None
            self.mats_env[new_idx].set_tensor(old_ts.clone())

        # Store cls_pos for convenience
        self.cls_pos = cls_pos

        # TODO: Add possibility for different embeddings for multi-label
        basis_embs = torch.stack([
            tk.embeddings.basis(torch.tensor(i), dim=self.num_cls).to(dtype=torch.float)
            for i in range(self.num_cls)
        ])

        # Register as buffer (not a Parameter, not trainable)
        self.register_buffer("cls_embs", basis_embs)

        # ParamNode -> Node, doesn't work with stacking
        self.mats_env[self.cls_pos] = self.mats_env[self.cls_pos].parameterize(parametrize)
        

    def forward(self, data, cls, *args, **kwargs):
        assert cls in range(self.num_cls), "not that many classes"
        assert data.shape[1] == self.n_features, "Input feature count mismatch"

        cls_emb = self.cls_embs[cls] # type: ignore
        aux_tensor = torch.einsum("c, lcr, rim -> lim", 
                                  cls_emb, 
                                  self.cls_tensor, 
                                  self.R_tensor)

             
        self.mats_env[self.cls_pos].tensor = aux_tensor

        return super().forward(data=data, *args, **kwargs)
    

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------ENSEMBLE MPS METHOD---------------------------------------------------------------------------------------------
#--------I MAY SWITCH TO THIS METHOD IF IT IS BETTER---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# Instead of a single MPS with a central cls tensor, one could train an ensemble of tensors
# interacting with each other only through the (classification) loss
# Conditioning implies what?


# TODO: Implement ensemble method