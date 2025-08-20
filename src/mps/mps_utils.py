import torch
import tensorkrowch as tk
from typing import Union, Sequence, Callable, Dict
import sampling
from torch.utils.data import DataLoader
import torch.nn as nn

# TODO: Change name to just born or something
def born_sequential(mps: tk.models.MPS, 
                    embs: dict | torch.Tensor)-> torch.Tensor: # very flexible, not parallizable
    """ 
    Applies Born rule to MPS contracted with MPS.
    Allows both for sequential and parallel contraction of MPS.
    Parameters
    ----------
    mps:    tk.models.MPS
        instance defining the prob amplitude
    embs:   embedded input
        dict of batch_size x phys_dim tensors or batch_size x n_feat x phys_dim tensor
    
    Returns
    -------
    tensor
        probability distribution(s)
    """

    
    if isinstance(embs, dict):
        mps.in_features = [i for i in embs.keys()]
        # input a list of tensors -> inefficient, if clauses could make this more general
        in_tensors = [embs[i] for i in mps.in_features]

        
        if len(in_tensors) < mps.n_features:
            return torch.diagonal(mps(in_tensors, marginalize_output=True))
        else:
            return torch.square(mps.forward(data=in_tensors))
    
    elif isinstance(embs, torch.Tensor): # embs=tensor (parallizable), assume mps.out_feature = [cls_pos] globally
        return torch.square(mps.forward(data=embs))
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Sampling routines using mps------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----Currenty on purely sequential implementation-------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: Add documentation, is this at the interface or implementation level?
# TODO: Think about moving away from dictionaries
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
        
        # Later, include an if clause here to inclue other sampling methods
        samples[site] = sampling.sss_sampling(p, input_space[site])
        
        embs[i] = embedding[site](samples[site], phys_dim[site])[:, None, :].expand(-1, num_bins, -1).reshape(num_samples*num_bins, -1)
        site += 1
    return samples

# TODO: Add documentation
# TODO: Think about moving away from dictionaries
def mps_sampling(   mps: tk.models.MPS,
                    input_space: Union[torch.Tensor, Sequence[torch.Tensor]], # need to have the same number of bins
                    embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                    num_samples: int, # per class
                    cls_pos: int,
                    cls_embs: Sequence[torch.Tensor], # one embedding per class,
                    device: torch.device):
    
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

# TODO: Add documentation
def batch_sampling_mps( mps: tk.models.MPS,
                        input_space: Union[torch.Tensor, Sequence[torch.Tensor]], # need to have the same number of bins
                        embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                        batch_size: int, # will be num_samples in mps_sampling
                        cls_pos: int,
                        cls_embs: Sequence[torch.Tensor], # one embedding per class,
                        total_n_samples: int, # total number of samples per class, ought to be divisible by batch_size
                        device: torch.device,
                        save_classes = False, # only needed for visualisation
):
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
            
        )
        samples = samples + [batch]
        if save_classes == True:
            batch_label = torch.concat([torch.full((batch_size,), cls, dtype=torch.long) for cls in range(len(cls_embs))])
            labels = labels + [batch_label]
    
    samples = torch.concat(samples)
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

# TODO: Add documentation
def mps_acc_eval(mps: tk.models.MPS, # relies on mps.out_features = [cls_pos]
                    loaders: Dict[str, DataLoader],
                    split: str,
                    device: torch.device):
    mps.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, t in loaders[split]:
            X, t = X.to(device), t.to(device)
            p = born_sequential(mps, X)
            _, preds = torch.max(p, 1)

            # Counting correct predictions
            correct += (preds == t).sum().item()
            total += t.size(0)
    acc = correct / total
    return acc

# TODO: Add documentation
def _mps_cls_train_step(mps: tk.models.MPS, # expect mps.out_features = [cls_pos]
                       loader: DataLoader,
                       device: torch.device,
                       optimizer: torch.optim.Optimizer,
                       loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] # expects score, not probs
                       ) -> list:
    train_loss = []
    mps.train()
    for X, t in loader:
        X, t = X.to(device), t.to(device)
        p = born_sequential(mps, X)
        # Computing Scores for loss, depending on model type
        score = torch.log(p)
        # Computing loss
        loss = loss_fn(score, t)
        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

# TODO: Add documentation
# TODO: Think about logging different or more quantities
def disr_train_mps( mps: tk.models.MPS,
                    loaders: Dict[str, DataLoader], #loads embedded inputs and labels
                    optimizer: torch.optim.Optimizer,
                    max_epoch: int,
                    patience: int,
                    cls_pos: int,
                    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    device: torch.device,
                    title: str | None = None,
                    goal_acc: float | None = None,
                    print_early_stop = True,
                    print_updates = True
                    ):
    val_accuracy = []
    train_loss = []
    best_acc = 0.0
    best_tensors = []  # Container for best model parameters

    # Prepare training
    mps.unset_data_nodes()
    mps.reset()
    mps.out_features = [cls_pos]

    phys_dim = mps.phys_dim[cls_pos-1]
    mps.trace(torch.zeros(1, len(mps.in_features), phys_dim).to(device))
        
    for i in range(max_epoch):
        # Training
        train_loss = train_loss + _mps_cls_train_step(mps, loaders['train'], 
                                  device, optimizer, loss_fn)

        # Validation
        acc = mps_acc_eval(mps, loaders, 'valid', device)
        val_accuracy.append(acc)

        # Progress tracking and best model update
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_tensors = mps.tensors
        else:
            patience_counter += 1

        # Early stopping via patience
        if print_early_stop and (patience_counter > patience):
            print(f"Early stopping via patience at epoch {i}")
            break
        if print_early_stop and (goal_acc is not None) and (goal_acc < best_acc):
            print(f"Early stopping via goal acc at epoch {i}")
            break
        # Update report
        if (i == 0) and (title is not None):
            print('')
            print(f'Training of {title} dataset:')
            print('')  
        elif print_updates and ((i+1) % 10 == 0):
            print(f"Epoch {i+1}: val_accuracy={acc:.4f}")

    return best_tensors, train_loss, val_accuracy



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------Other mps methods-------------------------------------------------------------------------------------------------------------------------------------
#------May or may not be used----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Write a function that for a given MPS (Layer) returns class embeddings

# TODO: Write a function that gives input potential form of input embeddings


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