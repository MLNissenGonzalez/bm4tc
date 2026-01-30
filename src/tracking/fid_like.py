import torch
import torch.nn as nn
import numpy as np
from typing import *
from torch.autograd import Function
import scipy.linalg
import src.utils.schemas as schemas
import logging
logger = logging.getLogger(__name__)

# FID like metric to judge generative capability.

# --- Matrix square root implementation for pytorch,
# from https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py ---

class MatrixSquareRoot(Function):
    """
    Custom PyTorch autograd function to compute the matrix square root.

    Computes X = sqrt(A) such that X @ X ≈ A. Backpropagation uses the
    Sylvester equation to compute the gradient w.r.t. the input matrix.

    Notes
    -----
    - Forward pass uses `scipy.linalg.sqrtm`.
    - Backward pass solves the Sylvester equation: sqrtm @ dX + dX @ sqrtm = d(sqrtm).
    - Inputs are assumed to be square matrices.
    - Input and output tensors are on the same device and dtype as the input.

    Usage
    -----
    >>> X = torch.randn(3, 3)
    >>> sqrtX = MatrixSquareRoot.apply(X)
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float64)
            gm = grad_output.detach().cpu().numpy().astype(np.float64)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def mean_n_cov(data: torch.FloatTensor):
    """
    Compute the mean vector and covariance matrix of a dataset.

    Parameters
    ----------
    data : torch.FloatTensor
        Tensor of shape (N, d) where N is the number of samples and d the feature dimension.

    Returns
    -------
    mu : torch.Tensor
        Mean vector of shape (d,).
    cov : torch.Tensor
        Covariance matrix of shape (d, d), computed as torch.cov(data.T).
    """
    mu = data.mean(dim=0)
    cov = torch.cov(data.T)
    return mu, cov


class FIDLike(nn.Module):
    """
    FID-like metric for evaluating the quality of generated samples.

    Measures the similarity between the distributions of real and generated samples
    using a Gaussian approximation: differences in mean and covariance, including
    a matrix square root term. Similar to the Fréchet Inception Distance (FID) in spirit.

    Parameters
    ----------
    eps : float
        Small regularization constant added to the diagonal of covariance matrices
        for numerical stability.

    Methods
    -------
    lazy_forward(mu_r, cov_r, generated)
        Computes the FID-like score given precomputed real mean/covariance and generated samples.
    forward(real, generated)
        Computes the FID-like score given raw real and generated samples.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def lazy_forward(self, mu_r: torch.Tensor, cov_r: torch.Tensor, generated: torch.Tensor):
        mu_g, cov_g = mean_n_cov(generated)

        # Regularize for numerical stability
        eye = torch.eye(cov_r.shape[0], device=cov_r.device)
        cov_r = cov_r + self.eps * eye
        cov_g = cov_g + self.eps * eye

        # Mean difference
        diff_mu = torch.sum((mu_r - mu_g) ** 2)

        # Matrix square root term using custom function
        covmean = sqrtm(cov_r @ cov_g)

        # Ensure real part (numerical precision can introduce tiny imaginary parts)
        if torch.is_complex(covmean):
            covmean = covmean.real

        diff_cov = torch.trace(cov_r + cov_g - 2 * covmean)

        return diff_mu + diff_cov

    def forward(self, real, generated):
        mu_r, cov_r = mean_n_cov(real)
        return self.lazy_forward(mu_r, cov_r, generated)
    
from src.data.handler import DataHandler

class FIDEvaluation:
    def __init__(
            self,
            cfg: schemas.Config,
            datahandler: DataHandler,
            device: torch.device,
    ):
        self.toEval = False
        self.samp_cfg = cfg.tracking.sampling
        self.device = device

        # Threshold condition: small data_dim => feasible to compute full covariance
        if datahandler.data_dim < 1e2:
            self.toEval = True
            self.stat_r = list(zip(datahandler.means, datahandler.covs))
            self.fid_like = FIDLike()
        else:
            logger.info("FID evaluation skipped: data_dim too large for classwise covariance computation.")

    def evaluate(
            self,
            synths: torch.Tensor
    ) -> float:
        """
        Evaluate FID-like distance between real and synthetic samples.

        Parameters
        ----------
        synths : torch.Tensor
            Synthetic samples per class, shape (n_samples_per_class, n_classes, data_dim)
        device : torch.device, optional
            Device for temporary computation (defaults to same as initialization)
        """
        if not self.toEval:
            raise ValueError("FID too expensive for this dataset (skipped).")

        device = self.device
        fid_values = []

        for c in range(synths.shape[1]):
            gen = synths[:, c, :].to(device)
            mu_r, cov_r = self.stat_r[c]
            mu_r, cov_r = mu_r.to(device), cov_r.to(device)

            fid_val = self.fid_like.lazy_forward(mu_r, cov_r, gen)
            fid_values.append(fid_val.cpu())  # move to CPU to safely stack later

        fid = torch.mean(torch.stack(fid_values)).item()
        return fid

