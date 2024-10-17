from abc import ABC, abstractmethod
from typing import Callable, Optional, List
import numpy as np

import torch
from torch import Tensor, nn
from torch.func import grad_and_value, grad, vmap
from torch.distributions import Normal
from numpy.typing import ArrayLike
from pdediff.sde import VPSDE
from pdediff.utils.data_preprocessing import append_zeros


######################################################
##################  Likelihood  ######################
######################################################


class Likelihood(ABC, nn.Module):
    """p(y|x_0) = p(y|A(x))"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], mask: Tensor = None):
        super().__init__()
        self.register_buffer("y", y)
        self.A = A
        self.mask = mask

    def err(self, x: Tensor) -> Tensor:
        if self.mask is not None:
            obs = self.A(x, self.mask.to(x.device))
        else:
            obs = self.A(x)

        assert obs.shape == self.y.shape, f"Obs shape {obs.shape} different to y shape {self.y.shape}"
        return self.y.to(x.device) - obs

    def set_observation(self, y: Tensor, mask: Tensor = None):
        self.y.copy_(y)
        if mask is not None:
            self.mask = mask



class Gaussian(Likelihood):
    """p(y|x_0) = N(y|A(x), std^2)"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], std: float = 1.0, mask: Tensor = None):
        super().__init__(y, A, mask)
        self.std = std

    def sample(self, shape=()) -> Tensor:
        return Normal(self.y, self.std).rsample(shape)


######################################################
################## Guidance term #####################
######################################################


class GuidedScore(ABC, nn.Module):
    def __init__(self, sde: VPSDE, likelihoods: Likelihood):
        super().__init__()
        self.likelihoods = likelihoods
        self.sde = sde

    def get_sde(self, shape) -> VPSDE:
        return self.sde.__class__(self, shape=shape)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class SDA(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps
    
class DPS(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            errs = []
            for likelihood in zip(self.likelihoods):
                errs.append((likelihood.err(x_)).square().sum())

        for err, gamma in zip(errs, self.gammas):
            s, = torch.autograd.grad(err, x, retain_graph=True)
            s = -s / gamma / err.sqrt()
            eps = eps - sigma * s

        return eps

class VideoDiff(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps

class PGDM(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = likelihood.std ** 2 + (sigma) ** 2 / (mu**2 + sigma**2)
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps